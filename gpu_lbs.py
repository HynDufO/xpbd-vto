# The MIT License (MIT)
# Copyright (c) 2022 NVIDIA
# www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# control:

# 'p': toggle paused
# 'h': toggle hidden
# 'c': solve type coloring hybrid
# 'j': solve type Jacobi
# 'r': reset state
# 'w' 's' 'a' 'd' 'e' 'q' : camera control
# left mouse view
# middle mouse pan
# right mouse orbit
# shift mouse interact

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import warp as wp
import math
import time
import heapq

wp.init()

import numpy as np
import pickle


class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def set_params(self, pose=None, beta=None, trans=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts

  def getG(self):
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    return G

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
          )
        )
      )
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_to_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

class ClothLBS:
    def __init__(self, smpl, vertices, faces, corr_path):
        self.ori_verts = np.array(vertices)
        self.verts = np.array(vertices)
        self.faces = np.array(faces)
        self.weights = np.zeros((len(self.verts), 24))
        with open(corr_path, 'r') as f:
            for line in f:
                u, v = map(int, line.split())
                self.weights[u] = smpl.weights[v]
    def update(self, smpl):
        G = smpl.getG()
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((self.ori_verts, np.ones([self.verts.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v

np.random.seed(9603)
smpl = SMPLModel('./model.pkl')
beta = np.array([
    -0.6233266592025757,
    2.5117039680480957,
    -0.07857337594032288,
    1.7532191276550293,
    -1.8665361404418945,
    -0.5116085410118103,
    1.2272087335586548,
    0.5566101670265198,
    0.7727207541465759,
    1.4483420968055725,
])
smpl.set_params(beta=beta)
# ---------------------------------------------

targetFps = 70
numSubsteps = 30
timeStep = 1.0 / 70.0
# timeStep = 3.0
gravity = wp.vec3(0.0, -9.8, 0.0)
velMax = 1.5
paused = False
hidden = False
frameNr = 0

# 0 Coloring
# 1 Jacobi
solveType = 0
jacobiScale = 0.2

clothNumX = 250
clothNumY = 250
clothY = 2.2
clothSpacing = 0.01
sphereCenter = wp.vec3(0.0, 1, 0.0)
sphereRadius = 0.13

# ---------------------------------------------

@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)

class Cloth:

    @wp.kernel
    def computeRestLengths(
            pos: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):
        cNr = wp.tid()
        p0 = pos[constIds[2 * cNr]]
        p1 = pos[constIds[2 * cNr + 1]]
        restLengths[cNr] = wp.length(p1 - p0)

    @wp.kernel
    def computeRestLengthsClothBody(
            clothPos: wp.array(dtype = wp.vec3),
            bodyPos: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):
        cNr = wp.tid()
        p0 = clothPos[constIds[2 * cNr]]
        p1 = bodyPos[constIds[2 * cNr + 1]]
        restLengths[cNr] = wp.length(p1 - p0) + 0.0005

    # -----------------------------------------------------
    def __init__(self, yOffset, numX, numY, spacing, sphereCenter, sphereRadius, objFilePath=None, bodyOBJFilePath=None, constraintsFilePath=None):
        if objFilePath != None:
            assert(constraintsFilePath != None)
            self.initWithOBJ(objFilePath, bodyOBJFilePath, constraintsFilePath)
            return
        device = "cpu"

        self.dragParticleNr = -1
        self.dragDepth = 0.0
        self.dragInvMass = 0.0
        self.renderParticles = []
        
        self.sphereCenter = sphereCenter
        self.sphereRadius = sphereRadius

        if numX % 2 == 1:
            numX = numX + 1
        if numY % 2 == 1:
            numY = numY + 1

        self.spacing = spacing
        self.numParticles = (numX + 1) * (numY + 1)
        pos = np.zeros(3 * self.numParticles)
        normals = np.zeros(3 * self.numParticles)
        invMass = np.zeros(self.numParticles)

        for xi in range(numX + 1):
            for yi in range(numY + 1):
                id = xi * (numY + 1) + yi
                pos[3 * id] = (-numX * 0.5 + xi) * spacing
                pos[3 * id + 1] = yOffset
                pos[3 * id + 2] = (-numY * 0.5 + yi) * spacing
                invMass[id] = 1.0
                # if yi == numY and (xi == 0 or xi == numX):
                #     invMass[id] = 0.0
                #     self.renderParticles.append(id)

        self.pos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.prevPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.restPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.invMass = wp.array(invMass, dtype = float, device = "cuda")
        self.corr = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.vel = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.normals = wp.array(normals, dtype = wp.vec3, device = "cuda")

        self.hostInvMass = wp.array(invMass, dtype = float, device = "cpu")
        self.hostPos = wp.array(pos, dtype = wp.vec3, device = "cpu")
        self.hostNormals = wp.array(normals, dtype = wp.vec3, device = "cpu")

        # constraints

        self.passSizes = [
            (numX + 1) * math.floor(numY / 2),
            (numX + 1) * math.floor(numY / 2),
            math.floor(numX / 2) * (numY + 1),
            math.floor(numX / 2) * (numY + 1),
            2 * numX * numY + (numX + 1) * (numY - 1) + (numY + 1) * (numX - 1)
            ]
        self.passIndependent = [
            True, True, True, True, False
        ]

        self.numDistConstraints = 0
        for passSize in self.passSizes:            
            self.numDistConstraints = self.numDistConstraints + passSize

        distConstIds = np.zeros(2 * self.numDistConstraints, dtype = wp.int32)

        # stretch constraints

        i = 0
        for passNr in range(2):
            for xi in range(numX + 1):
                for yi in range(math.floor(numY / 2)):
                    distConstIds[2 * i] = xi * (numY + 1) + 2 * yi + passNr
                    distConstIds[2 * i + 1] = xi * (numY + 1) + 2 * yi + passNr + 1
                    i = i + 1

        for passNr in range(2):
            for xi in range(math.floor(numX / 2)):
                for yi in range(numY + 1):
                    distConstIds[2 * i] = (2 * xi + passNr) * (numY + 1) + yi
                    distConstIds[2 * i + 1] = (2 * xi + passNr + 1) * (numY + 1) + yi
                    i = i + 1

        # shear constraints

        for xi in range(numX):
            for yi in range(numY):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 1) * (numY + 1) + yi + 1
                i = i + 1
                distConstIds[2 * i] = (xi + 1) * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 1
                i = i + 1

        # bending constraints

        for xi in range(numX + 1):
            for yi in range(numY - 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 2
                i = i + 1

        for xi in range(numX - 1):
            for yi in range(numY + 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 2) * (numY + 1) + yi                
                i = i + 1

        self.distConstIds = wp.array(distConstIds, dtype = wp.int32, device = "cuda")
        self.constRestLengths = wp.zeros(self.numDistConstraints, dtype = float, device = "cuda")

        wp.launch(kernel = self.computeRestLengths,
                inputs = [self.pos, self.distConstIds, self.constRestLengths], 
                dim = self.numDistConstraints,  device = "cuda")

        # tri ids

        self.numTris = 2 * numX * numY
        self.hostTriIds = np.zeros(3 * self.numTris, dtype = np.int32)

        i = 0
        for xi in range(numX):
            for yi in range(numY):
                id0 = xi * (numY + 1) + yi
                id1 = (xi + 1) * (numY + 1) + yi
                id2 = (xi + 1) * (numY + 1) + yi + 1
                id3 = xi * (numY + 1) + yi + 1

                self.hostTriIds[i] = id0
                self.hostTriIds[i + 1] = id1
                self.hostTriIds[i + 2] = id2

                self.hostTriIds[i + 3] = id0
                self.hostTriIds[i + 4] = id2
                self.hostTriIds[i + 5] = id3

                i = i + 6

        self.triIds = wp.array(self.hostTriIds, dtype = wp.int32, device = "cuda")

        self.triDist = wp.zeros(self.numTris, dtype = float, device = "cuda")
        self.hostTriDist = wp.zeros(self.numTris, dtype = float, device = "cpu")

        print(str(self.numTris) + " triangles created")
        print(str(self.numDistConstraints) + " distance constraints created")
        print(str(self.numParticles) + " particles created")

    def readOBJFile(self, file_path):
        vertices = []
        triangles = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):  # vertex
                    vertex = list(map(float, line.strip().split()[1:]))
                    vertices.append(vertex)
                elif line.startswith('f '):  # face (triangle)
                    face_vertices = [int(v.split('/')[0]) - 1 for v in line.strip().split()[1:]]
                    triangles.append(face_vertices)
        return vertices, triangles

    def initWithOBJ(self, objFilePath, bodyOBJFilePath, constraintsFilePath):

        self.dragParticleNr = -1
        self.dragDepth = 0.0
        self.dragInvMass = 0.0
        self.renderParticles = []
        self.renderParticles1 = []
        self.renderParticles2 = []
        
        self.adjIds = [0] * (20 * 40000)
        self.adjIdsTri = [0] * (20 * 40000)

        vertices, triangles = self.readOBJFile(objFilePath)
        self.clothLBS = ClothLBS(smpl, vertices, triangles, "corr_weights.txt")
        bodyVertices, bodyTriangles = smpl.verts, smpl.faces 
        self.numParticles = len(vertices)
        self.numBodyParticles = len(bodyVertices)
        pos = np.zeros(3 * self.numParticles)
        bodyPos = np.zeros(3 * len(bodyVertices))
        normals = np.zeros(3 * self.numParticles)
        invMass = np.zeros(self.numParticles)

        for i in range(self.numParticles):
            pos[3 * i] = vertices[i][0]
            pos[3 * i + 1] = vertices[i][1]
            pos[3 * i + 2] = vertices[i][2]
            invMass[i] = 1

        for i in range(len(bodyVertices)):
            bodyPos[3 * i] = bodyVertices[i][0]
            bodyPos[3 * i + 1] = bodyVertices[i][1]
            bodyPos[3 * i + 2] = bodyVertices[i][2]

        self.pos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.prevPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.restPos = wp.array(pos, dtype = wp.vec3, device = "cuda")
        self.invMass = wp.array(invMass, dtype = float, device = "cuda")
        self.corr = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.vel = wp.array(np.zeros(3 * self.numParticles), dtype = wp.vec3, device = "cuda")
        self.normals = wp.array(normals, dtype = wp.vec3, device = "cuda")

        self.hostInvMass = wp.array(invMass, dtype = float, device = "cpu")
        self.hostPos = wp.array(pos, dtype = wp.vec3, device = "cpu")
        self.hostNormals = wp.array(normals, dtype = wp.vec3, device = "cpu")
        self.bodyPos = wp.array(bodyPos, dtype = wp.vec3, device = "cuda")
        self.hostBodyPos = wp.array(bodyPos, dtype = wp.vec3, device = "cpu")

        # constraints
        distConstIds = np.zeros(1, dtype = wp.int32)
        self.passSizes = []
        self.passIndependent = []

        with open(constraintsFilePath, "r") as file:
            lines = file.readlines()
            numV, numTri, numCons = map(int, lines[0].split())
            assert(numV == len(vertices))
            assert(numTri == len(triangles))
            self.numDistConstraints = numCons
            distConstIds = np.zeros(2 * self.numDistConstraints, dtype = wp.int32)
            id = 1
            ptr = 0
            for _ in range(1):
                numGroups = int(lines[id])
                lst = False
                id += 1
                for i in range(numGroups):
                    groupSize = int(lines[id])
                    if lst and len(self.passSizes) > 0 and groupSize < 3000:
                        self.passSizes[-1] += groupSize
                        self.passIndependent[-1] = False
                    else:
                        self.passSizes.append(groupSize)
                        self.passIndependent.append(True)
                    id += 1
                    for _ in range(groupSize):
                        u, v = map(int, lines[id].split())
                        distConstIds[ptr] = u
                        distConstIds[ptr + 1] = v
                        ptr += 2
                        id += 1
                    if groupSize < 3000:
                        lst = True
                    else:
                        lst = False
        assert(ptr == 2 * self.numDistConstraints)
        print(self.passSizes)
        print(self.passIndependent)

        self.distConstIds = wp.array(distConstIds, dtype = wp.int32, device = "cuda")
        self.constRestLengths = wp.zeros(self.numDistConstraints, dtype = float, device = "cuda")

        distConstIdsClothBody = np.zeros(1, dtype = wp.int32)
        with open("cloth_body_constraints.txt", "r") as file:
            lines = file.readlines()
            numClothBodyConstraints = int(lines[0])
            self.numClothBodyConstraints = numClothBodyConstraints
            distConstIdsClothBody = np.zeros(2 * numClothBodyConstraints, dtype = wp.int32)
            for i in range(numClothBodyConstraints):
                clothId, bodyId = map(int, lines[i + 1].split())
                distConstIdsClothBody[2 * i] = clothId
                distConstIdsClothBody[2 * i + 1] = bodyId

        self.distConstIdsClothBody = wp.array(distConstIdsClothBody, dtype = wp.int32, device = "cuda")
        self.constRestLengthsClothBody = wp.zeros(len(distConstIdsClothBody), dtype = float, device = "cuda")

        wp.launch(kernel = self.computeRestLengths,
                inputs = [self.pos, self.distConstIds, self.constRestLengths], 
                dim = self.numDistConstraints,  device = "cuda")
        wp.launch(kernel = self.computeRestLengthsClothBody,
                inputs = [self.pos, self.bodyPos, self.distConstIdsClothBody, self.constRestLengthsClothBody], 
                dim = len(distConstIdsClothBody),  device = "cuda")


        # tri ids

        self.numTris = len(triangles)
        self.hostTriIds = np.zeros(3 * self.numTris, dtype = np.int32)
        for i in range(len(triangles)):
            self.hostTriIds[3 * i] = triangles[i][0]
            self.hostTriIds[3 * i + 1] = triangles[i][1]
            self.hostTriIds[3 * i + 2] = triangles[i][2]

        self.bodyTriIds = np.zeros(3 * len(bodyTriangles), dtype = np.int32)
        for i in range(len(bodyTriangles)):
            self.bodyTriIds[3 * i] = bodyTriangles[i][0]
            self.bodyTriIds[3 * i + 1] = bodyTriangles[i][1]
            self.bodyTriIds[3 * i + 2] = bodyTriangles[i][2]

        self.triIds = wp.array(self.hostTriIds, dtype = wp.int32, device = "cuda")
        self.bodyTriIdsCuda = wp.array(self.bodyTriIds, dtype = wp.int32, device = "cuda")

        self.triDist = wp.zeros(self.numTris, dtype = float, device = "cuda")
        self.hostTriDist = wp.zeros(self.numTris, dtype = float, device = "cpu")

        start_time = time.time()
        # self.numCollisions = self.queryAllCloseBodyParticles(self.hostTriIds, pos, bodyPos, 5)
        self.numColClothToBody, self.numColBodyToCloth = self.readTriPointPairs("tri_point_pairs.txt", self.hostTriIds, self.bodyTriIds)
        end_time = time.time()
        print("Elapsed time:", end_time - start_time, "seconds")
        print("Num collisions =", self.numColClothToBody, self.numColBodyToCloth)
        print(str(self.numTris) + " triangles created")
        print(str(self.numDistConstraints + self.numClothBodyConstraints) + " distance constraints created")
        print(str(self.numParticles) + " particles created")

    # ----------------------------------
    @wp.kernel
    def addNormals(
            pos: wp.array(dtype = wp.vec3),
            triIds: wp.array(dtype = wp.int32),
            normals: wp.array(dtype = wp.vec3)):
        triNr = wp.tid()

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]
        normal = wp.cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
        wp.atomic_add(normals, id0, normal)
        wp.atomic_add(normals, id1, normal)
        wp.atomic_add(normals, id2, normal)

    @wp.kernel
    def normalizeNormals(
            normals: wp.array(dtype = wp.vec3)):

        pNr = wp.tid()
        normals[pNr] = wp.normalize(normals[pNr])

    def updateMesh(self):
        self.normals.zero_()
        wp.launch(kernel = self.addNormals, inputs = [self.pos, self.triIds, self.normals], dim = self.numTris, device = "cuda")
        wp.launch(kernel = self.normalizeNormals, inputs = [self.normals], dim = self.numParticles, device = "cuda")
        wp.copy(self.hostNormals, self.normals)

    # ----------------------------------

    @wp.kernel
    def integrate(
            dt: float,
            gravity: wp.vec3,
            invMass: wp.array(dtype = float),
            prevPos: wp.array(dtype = wp.vec3),
            pos: wp.array(dtype = wp.vec3),
            vel: wp.array(dtype = wp.vec3),
            sphereCenter: wp.vec3,
            sphereRadius: float):

        pNr = wp.tid()

        prevPos[pNr] = pos[pNr]
        if invMass[pNr] == 0.0:
            return
        vel[pNr] = vel[pNr] + gravity * dt
        pos[pNr] = pos[pNr] + vel[pNr] * dt
        
        # collisions
        
        thickness = 0.001
        friction = 0.01

        d = wp.length(pos[pNr] - sphereCenter)
        if d < (sphereRadius + thickness):
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            r = p - sphereCenter
            d = wp.length(r)            
            pos[pNr] = sphereCenter + r * ((sphereRadius + thickness) / d)
            
        p = pos[pNr]
        if p[1] < thickness:
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            pos[pNr] = wp.vec3(p[0], thickness, p[2])

    @wp.kernel
    def update_body_pos(
        pos: wp.array(dtype = wp.vec3),
        delta: wp.array(dtype = wp.vec3)):

        pNr = wp.tid()
        pos[pNr] += delta[pNr] 

    @wp.kernel
    def integrate_gravity(
            dt: float,
            vMax: float,
            gravity: wp.vec3,
            invMass: wp.array(dtype = float),
            prevPos: wp.array(dtype = wp.vec3),
            pos: wp.array(dtype = wp.vec3),
            vel: wp.array(dtype = wp.vec3)):

        pNr = wp.tid()

        prevPos[pNr] = pos[pNr]
        if invMass[pNr] == 0.0:
            return
        # enforce velocity limit to prevent instability
        v1 = vel[pNr] + gravity * dt
        v1_mag = wp.length(v1)
        if v1_mag > vMax:
            v1 *= vMax / v1_mag
        pos[pNr] = pos[pNr] + v1 * dt
        vel[pNr] = v1

    @wp.kernel
    def integrate_body_collisions(
            dt: float,
            invMass: wp.array(dtype = float),
            pos: wp.array(dtype = wp.vec3),
            forces: wp.array(dtype = wp.vec3)):

        pNr = wp.tid()

        if invMass[pNr] == 0.0:
            return
        c = forces[pNr] * invMass[pNr] * dt
        pos[pNr] = pos[pNr] + c * dt
        # vel[pNr] = vel[pNr] + c

    @wp.kernel
    def collide_cloth_tri_body_particle_fast(
        adjIds: wp.array(dtype = wp.int32),
        adjIdsTri: wp.array(dtype = wp.int32),
        pos: wp.array(dtype = wp.vec3),
        bodyPos: wp.array(dtype = wp.vec3),
        clothTriIds: wp.array(dtype = wp.int32),
        f: wp.array(dtype = wp.vec3)
    ):
        tid = wp.tid()
        triNo = adjIdsTri[tid] 
        particleNo = adjIds[tid]

        particle = bodyPos[particleNo]
        i, j, k = clothTriIds[3 * triNo], clothTriIds[3 * triNo + 1], clothTriIds[3 * triNo + 2]
        p = pos[i]
        q = pos[j]
        r = pos[k]

        bary = triangle_closest_point_barycentric(p, q, r, particle)
        closest = p * bary[0] + q * bary[1] + r * bary[2]

        diff = particle - closest
        dist = wp.dot(diff, diff)
        n = wp.normalize(diff)
        c = wp.min(dist - 0.00025, 0.0)  # 0 unless within 0.01 of surface
        if c == 0:
            return
        fn = n * c * 8e6
        wp.atomic_add(f, i, fn * bary[0])
        wp.atomic_add(f, j, fn * bary[1])
        wp.atomic_add(f, k, fn * bary[2])
        
    @wp.kernel
    def collide_body_tri_cloth_particle_fast(
        adjIds: wp.array(dtype = wp.int32),
        adjIdsTri: wp.array(dtype = wp.int32),
        pos: wp.array(dtype = wp.vec3),
        bodyPos: wp.array(dtype = wp.vec3),
        clothTriIds: wp.array(dtype = wp.int32),
        f: wp.array(dtype = wp.vec3)
    ):
        tid = wp.tid()
        triNo = adjIdsTri[tid] 
        particleNo = adjIds[tid]

        particle = bodyPos[particleNo]
        i, j, k = clothTriIds[3 * triNo], clothTriIds[3 * triNo + 1], clothTriIds[3 * triNo + 2]
        p = pos[i]
        q = pos[j]
        r = pos[k]

        bary = triangle_closest_point_barycentric(p, q, r, particle)
        closest = p * bary[0] + q * bary[1] + r * bary[2]

        diff = particle - closest
        dist = wp.dot(diff, diff)
        n = wp.normalize(diff)
        c = wp.min(dist - 0.00025, 0.0)  # 0 unless within 0.01 of surface
        if c == 0:
            return
        fn = n * c * 1e7
        wp.atomic_sub(f, particleNo, fn)

    # ----------------------------------
    @wp.kernel
    def solveDistanceConstraints(
            solveType: wp.int32,
            firstConstraint: wp.int32,
            invMass: wp.array(dtype = float),
            pos: wp.array(dtype = wp.vec3),
            corr: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):

        cNr = firstConstraint + wp.tid()
        id0 = constIds[2 * cNr]
        id1 = constIds[2 * cNr + 1]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        if w == 0.0:
            return
        p0 = pos[id0]
        p1 = pos[id1]
        d = p1 - p0
        n = wp.normalize(d)
        l = wp.length(d)
        l0 = restLengths[cNr]
        dP = n * (l - l0) / w
        if solveType == 1:
            wp.atomic_add(corr, id0, w0 * dP)
            wp.atomic_sub(corr, id1, w1 * dP)
        else:
            wp.atomic_add(pos, id0, w0 * dP)
            wp.atomic_sub(pos, id1, w1 * dP)

    @wp.kernel
    def solveDistanceConstraintsClothBody(
            invMass: wp.array(dtype = float),
            pos: wp.array(dtype = wp.vec3),
            bodyPos: wp.array(dtype = wp.vec3),
            constIds: wp.array(dtype = wp.int32),
            restLengths: wp.array(dtype = float)):

        cNr = wp.tid()
        id0 = constIds[2 * cNr]
        id1 = constIds[2 * cNr + 1]
        w0 = invMass[id0]
        w = w0 + 1.0
        p0 = pos[id0]
        p1 = bodyPos[id1]
        d = p1 - p0
        n = wp.normalize(d)
        l = wp.length(d)
        l0 = restLengths[cNr]
        dP = n * (l - l0) / (w)
        wp.atomic_add(pos, id0, w0 * dP)

    # ----------------------------------
    @wp.kernel
    def addCorrections(
            pos: wp.array(dtype = wp.vec3),
            corr: wp.array(dtype = wp.vec3),
            scale: float):
        pNr = wp.tid()
        pos[pNr] = pos[pNr] + corr[pNr] * scale

    # ----------------------------------
    @wp.kernel
    def updateVel(
            dt: float,
            prevPos: wp.array(dtype = wp.vec3),
            pos: wp.array(dtype = wp.vec3),
            vel: wp.array(dtype = wp.vec3)):
        pNr = wp.tid()
        vel[pNr] = (pos[pNr] - prevPos[pNr]) / dt


    # ----------------------------------
    def getDist2(self, x, y, z, x1, y1, z1):
        return (x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)
    # ----------------------------------
    def readTriPointPairs(self, filename, triIds, triIds1):
        with open(filename, 'r') as file:
            n = int(file.readline().strip())  # Read the length of the arrays
            num = 0
            self.adjIds = [0] * n
            self.adjIdsTri = [0] * n
            for _ in range(n):
                idTri, id = map(int, file.readline().strip().split())
                self.adjIds[num] = id
                self.adjIdsTri[num] = idTri
                if idTri == 7135 and len(self.renderParticles1) == 0:
                    self.renderParticles1.append(triIds[3 * idTri])
                    self.renderParticles1.append(triIds[3 * idTri + 1])
                    self.renderParticles1.append(triIds[3 * idTri + 2])
                if idTri == 7135:
                    self.renderParticles2.append(self.adjIds[num])
                num += 1
            n1 = n
            n = int(file.readline().strip())  # Read the length of the arrays
            num = 0
            self.adjIds1 = [0] * n
            self.adjIdsTri1 = [0] * n
            for _ in range(n):
                idTri, id = map(int, file.readline().strip().split())
                self.adjIds1[num] = id
                self.adjIdsTri1[num] = idTri
                if idTri == 13637 and len(self.renderParticles1) == 3:
                    self.renderParticles2.append(triIds1[3 * idTri])
                    self.renderParticles2.append(triIds1[3 * idTri + 1])
                    self.renderParticles2.append(triIds1[3 * idTri + 2])
                if idTri == 13637:
                    self.renderParticles1.append(self.adjIds1[num])
                num += 1
            self.adjIdsCuda = wp.array(self.adjIds, dtype = wp.int32, device = "cuda")
            self.adjIdsTriCuda = wp.array(self.adjIdsTri, dtype = wp.int32, device = "cuda")
            self.adjIdsCuda1 = wp.array(self.adjIds1, dtype = wp.int32, device = "cuda")
            self.adjIdsTriCuda1 = wp.array(self.adjIdsTri1, dtype = wp.int32, device = "cuda")
            return n1, n

    # ----------------------------------
    def queryAllCloseBodyParticles(self, triIds, triPos, pos, numPointsEach):
        num = 0
        numTris = len(triIds) // 3
        numPoints = len(pos) // 3
        self.adjIds = [0] * (numPointsEach * numTris)
        self.adjIdsTri = [0] * (numPointsEach * numTris)
        for iTri in range(numTris):
            print(iTri)
            heap = []
            for iPoint in range(numPoints):
                sumDist2 = 0
                x, y, z = pos[3 * iPoint], pos[3 * iPoint + 1], pos[3 * iPoint + 2]
                for j in range(3):
                    p = triIds[3 * iTri + j]
                    dist2 = self.getDist2(triPos[3 * p], triPos[3 * p + 1], triPos[3 * p + 2], x, y, z)
                    sumDist2 += dist2
                heapq.heappush(heap, (-sumDist2, iPoint))
                if len(heap) > numPointsEach:
                    heapq.heappop(heap)
            assert(len(heap) == numPointsEach)
            for i in range(numPointsEach):
                self.adjIds[num] = heap[i][1]
                self.adjIdsTri[num] = iTri 
                if iTri == 12000:
                    self.renderParticles2.append(self.adjIds[num])
                num += 1
            if iTri == 12000:
                self.renderParticles1.append(triIds[3 * iTri])
                self.renderParticles1.append(triIds[3 * iTri + 1])
                self.renderParticles1.append(triIds[3 * iTri + 2])

        self.adjIdsCuda = wp.array(self.adjIds[:num], dtype = wp.int32, device = "cuda")
        self.adjIdsTriCuda = wp.array(self.adjIdsTri[:num], dtype = wp.int32, device = "cuda")
        return num

    # ----------------------------------
    def simulate(self):
        start_time = time.time()

        # Read 24x3 poses data from stdin
        line = sys.stdin.readline().strip()
        poses = np.array(list(map(float, line.split()))).reshape((24, 3))
        smpl.set_params(pose=poses)
        self.clothLBS.update(smpl)

        self.hostBodyPos = wp.array(smpl.verts, dtype = wp.vec3, device = "cpu")
        self.hostPos = wp.array(self.clothLBS.verts, dtype = wp.vec3, device = "cpu")
        end_time = time.time()
        print("Time each frame:", end_time - start_time, "seconds")
        print("Max FPS can be:", 1 / (end_time - start_time))

    # -------------------------------------------------
    def reset(self):
        self.vel.zero_()
        wp.copy(self.pos, self.restPos)

    # -------------------------------------------------
    @wp.kernel
    def raycastTriangle(
            orig: wp.vec3,
            dir: wp.vec3,
            pos: wp.array(dtype = wp.vec3),
            triIds: wp.array(dtype = wp.int32),
            dist: wp.array(dtype = float)):
        triNr = wp.tid()
        noHit = 1.0e6

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]              
        pNr = wp.tid()

        edge1 = pos[id1] - pos[id0]
        edge2 = pos[id2] - pos[id0]
        pvec = wp.cross(dir, edge2)
        det = wp.dot(edge1, pvec)

        if (det == 0.0):
            dist[triNr] = noHit
            return

        inv_det = 1.0 / det
        tvec = orig - pos[id0]
        u = wp.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            dist[triNr] = noHit
            return 

        qvec = wp.cross(tvec, edge1)
        v = wp.dot(dir, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            dist[triNr] = noHit
            return

        dist[triNr] = wp.dot(edge2, qvec) * inv_det

    # ------------------------------------------------
    def startDrag(self, orig, dir):
        
        wp.launch(kernel = self.raycastTriangle, inputs = [
            wp.vec3(orig[0], orig[1], orig[2]), wp.vec3(dir[0], dir[1], dir[2]), 
            self.pos, self.triIds, self.triDist], dim = self.numTris, device = "cuda")
        wp.copy(self.hostTriDist, self.triDist)

        pos = self.hostPos.numpy()
        self.dragDepth = 0.0

        dists = self.hostTriDist.numpy()
        minTriNr = np.argmin(dists)
        if dists[minTriNr] < 1.0e6:
            self.dragParticleNr = self.hostTriIds[3 * minTriNr]
            self.dragDepth = dists[minTriNr]
            invMass = self.hostInvMass.numpy()
            self.dragInvMass = invMass[self.dragParticleNr]
            invMass[self.dragParticleNr] = 0.0
            wp.copy(self.invMass, self.hostInvMass)

            pos = self.hostPos.numpy()
            dragPos = wp.vec3(
                orig[0] + self.dragDepth * dir[0], 
                orig[1] + self.dragDepth * dir[1], 
                orig[2] + self.dragDepth * dir[2])
            pos[self.dragParticleNr] = dragPos

            wp.copy(self.pos, self.hostPos)
        
    def drag(self, orig, dir):
        if self.dragParticleNr >= 0:
            pos = self.hostPos.numpy()
            dragPos = wp.vec3(
                orig[0] + self.dragDepth * dir[0], 
                orig[1] + self.dragDepth * dir[1], 
                orig[2] + self.dragDepth * dir[2])
            pos[self.dragParticleNr] = dragPos
            wp.copy(self.pos, self.hostPos)

    def endDrag(self):
        if self.dragParticleNr >= 0:
            invMass = self.hostInvMass.numpy()
            invMass[self.dragParticleNr] = self.dragInvMass
            wp.copy(self.invMass, self.hostInvMass)
            self.dragParticleNr = -1

    def render(self):

        # cloth

        twoColors = False

        glColor3f(1.0, 0.0, 0.0)
        glNormal3f(0.0, 0.0, -1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.hostPos.numpy())
        glNormalPointer(GL_FLOAT, 0, self.hostNormals.numpy())

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
  
        if twoColors:
            glCullFace(GL_FRONT)
            glColor3f(1.0, 1.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glCullFace(GL_BACK)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
        else:
            glDisable(GL_CULL_FACE)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glEnable(GL_CULL_FACE)

        # body
        glDisable(GL_LIGHTING);
        glColor3f(0.4, 0.4, 0.4)
        glVertexPointer(3, GL_FLOAT, 0, self.hostBodyPos.numpy())
        glDisable(GL_CULL_FACE)
        glColor3f(0.4, 0.4, 0.4)
        glDrawElementsui(GL_TRIANGLES, self.bodyTriIds)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # kinematic particles

        glColor3f(1.0, 1.0, 1.0)
        pos = self.hostPos.numpy()
        bodyPos = self.hostBodyPos.numpy()

        q = gluNewQuadric()

        if self.dragParticleNr >= 0:
            self.renderParticles.append(self.dragParticleNr)

        # for id in self.renderParticles:
        #     glPushMatrix()
        #     p = pos[id]
        #     glTranslatef(p[0], p[1], p[2])
        #     gluSphere(q, 0.005, 0, 0)
        #     glPopMatrix()

        if self.dragParticleNr >= 0:
            self.renderParticles.pop()

        for id in self.renderParticles1:
            glPushMatrix()
            p = pos[id]
            glTranslatef(p[0], p[1], p[2])
            gluSphere(q, 0.005, 40, 40)
            glPopMatrix()

        for id in self.renderParticles2:
            glPushMatrix()
            p = bodyPos[id]
            glTranslatef(p[0], p[1], p[2])
            glColor3f(0.0, 0.0, 1.0)
            gluSphere(q, 0.005, 40, 40)
            glPopMatrix()
            
        # sphere
        # glColor3f(0.8, 0.8, 0.8)
        #
        # glPushMatrix()
        # glTranslatef(self.sphereCenter[0], self.sphereCenter[1], self.sphereCenter[2])
        # gluSphere(q, self.sphereRadius, 40, 40)
        # glPopMatrix()

        gluDeleteQuadric(q)



# --------------------------------------------------------------------
# Demo Viewer
# --------------------------------------------------------------------

groundVerts = []
groundIds = []
groundColors = []
cloth = []

# -------------------------------------------------------
def initScene():
    global cloth
  
    # cloth = Cloth(clothY, clothNumX, clothNumY, clothSpacing, sphereCenter, sphereRadius)
    cloth = Cloth(clothY, clothNumX, clothNumY, clothSpacing, sphereCenter, sphereRadius, "starlight_tri.obj", "body.obj", "starlight_tri_constraints_groups_1.txt")

# --------------------------------
def showScreen():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ground plane

    # glColor3f(1.0, 1.0, 1.0)
    # glNormal3f(0.0, 1.0, 0.0)
    #
    # numVerts = math.floor(len(groundVerts) / 3)
    #
    # glVertexPointer(3, GL_FLOAT, 0, groundVerts)
    # glColorPointer(3, GL_FLOAT, 0, groundColors)
    #
    # glEnableClientState(GL_VERTEX_ARRAY)
    # glEnableClientState(GL_COLOR_ARRAY)
    # glDrawArrays(GL_QUADS, 0, numVerts)
    # glDisableClientState(GL_VERTEX_ARRAY)
    # glDisableClientState(GL_COLOR_ARRAY)

    # objects

    if not hidden:
        cloth.render()

    glutSwapBuffers()

# -----------------------------------
class Camera:
    def __init__(self):
        self.pos = wp.vec3(0.0, 1.0, 5.0)
        self.forward = wp.vec3(0.0, 0.0, -1.0)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.speed = 0.1
        self.keyDown = [False] * 256

    def rot(self, unitAxis, angle, v):
       q = wp.quat_from_axis_angle(unitAxis, angle)
       return wp.quat_rotate(q, v)

    def setView(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = viewport[2] - viewport[0]
        height = viewport[3] - viewport[1]

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, float(width) / float(height), 0.01, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt( 
            self.pos[0], self.pos[1], self.pos[2], 
            self.pos[0] + self.forward[0], self.pos[1] + self.forward[1], self.pos[2] + self.forward[2], 
            self.up[0], self.up[1], self.up[2])

    def lookAt(self, pos, at):
        self.pos = pos
        self.forward = wp.sub(at, pos)
        self.forward = wp.normalize(self.forward)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)

    def handleMouseTranslate(self, dx, dy):
        
        scale = wp.length(self.pos) * 0.001
        self.pos = wp.sub(self.pos, wp.mul(self.right, scale * float(dx)))
        self.pos = wp.add(self.pos, wp.mul(self.up, scale * float(dy)))

    def handleWheel(self, direction):
        self.pos = wp.add(self.pos, wp.mul(self.forward, direction * self.speed))

    def handleMouseView(self, dx, dy):
        scale = 0.005
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.forward = wp.normalize(self.forward)
        self.right = wp.cross(self.forward, self.up)
        self.right = wp.vec3(self.right[0], 0.0, self.right[2])
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)
        self.forward = wp.cross(self.up, self.right)
    
    def handleKeyDown(self, key):
        self.keyDown[ord(key)] = True

    def handleKeyUp(self, key):
        self.keyDown[ord(key)] = False

    def handleKeys(self):
        if self.keyDown[ord('+')]:
            self.speed = self.speed * 1.2
        if self.keyDown[ord('-')]:
            self.speed = self.speed * 0.8
        if self.keyDown[ord('w')]:
            self.pos = wp.add(self.pos, wp.mul(self.forward, self.speed))
        if self.keyDown[ord('s')]:
            self.pos = wp.sub(self.pos, wp.mul(self.forward, self.speed))
        if self.keyDown[ord('a')]:
            self.pos = wp.sub(self.pos, wp.mul(self.right, self.speed))
        if self.keyDown[ord('d')]:
            self.pos = wp.add(self.pos, wp.mul(self.right, self.speed))
        if self.keyDown[ord('e')]:
            self.pos = wp.sub(self.pos, wp.mul(self.up, self.speed))
        if self.keyDown[ord('q')]:
            self.pos = wp.add(self.pos, wp.mul(self.up, self.speed))

    def handleMouseOrbit(self, dx, dy, center):

        offset = wp.sub(self.pos, center)
        offset = [
            wp.dot(self.right, offset),
            wp.dot(self.forward, offset),
            wp.dot(self.up, offset)]

        scale = 0.01
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.up = self.rot(self.up, -dx * scale, self.up)
        self.up = self.rot(self.right, -dy * scale, self.up)

        self.right = wp.cross(self.forward, self.up)
        self.right = wp.vec3(self.right[0], 0.0, self.right[2])
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)
        self.forward = wp.cross(self.up, self.right)
        self.pos = wp.add(center, wp.mul(self.right, offset[0]))
        self.pos = wp.add(self.pos, wp.mul(self.forward, offset[1]))
        self.pos = wp.add(self.pos, wp.mul(self.up, offset[2]))

camera = Camera()

# ---- callbacks ----------------------------------------------------

mouseButton = 0
mouseX = 0
mouseY = 0
shiftDown = False

def getMouseRay(x, y):
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelMatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    projMatrix = glGetDoublev(GL_PROJECTION_MATRIX)

    y = viewport[3] - y - 1
    p0 = gluUnProject(x, y, 0.0, modelMatrix, projMatrix, viewport)
    p1 = gluUnProject(x, y, 1.0, modelMatrix, projMatrix, viewport)
    orig = wp.vec3(p0[0], p0[1], p0[2])
    dir = wp.sub(wp.vec3(p1[0], p1[1], p1[2]), orig)
    dir = wp.normalize(dir)
    return [orig, dir]

def mouseButtonCallback(button, state, x, y):
    global mouseX
    global mouseY
    global mouseButton
    global shiftDown
    global paused
    mouseX = x
    mouseY = y
    if state == GLUT_DOWN:
        mouseButton = button
    else:
        mouseButton = 0
    shiftDown = glutGetModifiers() & GLUT_ACTIVE_SHIFT
    if shiftDown:
        ray = getMouseRay(x, y)
        if state == GLUT_DOWN:
            cloth.startDrag(ray[0], ray[1])
            paused = False
    if state == GLUT_UP:
        cloth.endDrag()

def mouseMotionCallback(x, y):
    global mouseX
    global mouseY
    global mouseButton
    
    dx = x - mouseX
    dy = y - mouseY
    if shiftDown:
        ray = getMouseRay(x, y)
        cloth.drag(ray[0], ray[1])
    else:
        if mouseButton == GLUT_MIDDLE_BUTTON:
            camera.handleMouseTranslate(dx, dy)
        elif mouseButton == GLUT_LEFT_BUTTON:
            camera.handleMouseView(dx, dy)
        elif mouseButton == GLUT_RIGHT_BUTTON:
            camera.handleMouseOrbit(dx, dy, wp.vec3(0.0, 1.0, 0.0))

    mouseX = x
    mouseY = y        

def mouseWheelCallback(wheel, direction, x, y):
    camera.handleWheel(direction)

def handleKeyDown(key, x, y):
    camera.handleKeyDown(key)
    global paused
    global solveType
    global hidden
    if key == b'p':
        paused = not paused
    elif key == b'h':
        hidden = not hidden
    elif key == b'c':
        solveType = 0
    elif key == b'j':
        solveType = 1
    elif key == b'r':
        cloth.reset()

def handleKeyUp(key, x, y):
    camera.handleKeyUp(key)
    
def displayCallback():
    i = 0

prevTime = time.time()

def timerCallback(val):
    global prevTime
    global frameNr
    frameNr = frameNr + 1
    numFpsFrames = 30
    currentTime = time.perf_counter()

    if frameNr % numFpsFrames == 0:
        passedTime = currentTime - prevTime
        prevTime = currentTime
        fps = math.floor(numFpsFrames / passedTime)
        glutSetWindowTitle("Parallel cloth simulation " + str(fps) + " fps")
        print("Parallel cloth simulation " + str(fps) + " fps")
        print("Time:", passedTime)

    if not paused:
        cloth.simulate()

    cloth.updateMesh()
    showScreen()
    camera.setView()
    camera.handleKeys()

    elapsed_ms = (time.perf_counter() - currentTime) * 1000
    glutTimerFunc(max(0, math.floor((1000.0 / targetFps) - elapsed_ms)), timerCallback, 0)

# -----------------------------------------------------------

def setupOpenGL():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    ambientColor = [0.2, 0.2, 0.2, 1.0]
    diffuseColor = [0.8, 0.8 ,0.8, 1.0]
    specularColor = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientColor)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseColor)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularColor)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor)
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

    lightPosition = [10.0, 10.0 , 10.0, 0.0]
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)

    glEnable(GL_NORMALIZE)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)

    groundNumTiles = 30
    groundTileSize = 0.5

    global groundVerts
    global groundIds
    global groundColors

    groundVerts = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)
    groundColors = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype = float)

    squareVerts = [[0,0], [0,1], [1,1], [1,0]]
    r = groundNumTiles / 2.0 * groundTileSize

    for xi in range(groundNumTiles):
        for zi in range(groundNumTiles):
            x = (-groundNumTiles / 2.0 + xi) * groundTileSize
            z = (-groundNumTiles / 2.0 + zi) * groundTileSize
            p = xi * groundNumTiles + zi
            for i in range(4):
                q = 4 * p + i
                px = x + squareVerts[i][0] * groundTileSize
                pz = z + squareVerts[i][1] * groundTileSize
                groundVerts[3 * q] = px
                groundVerts[3 * q + 2] = pz
                col = 0.4
                if (xi + zi) % 2 == 1:
                    col = 0.8
                pr = math.sqrt(px * px + pz * pz)
                d = max(0.0, 1.0 - pr / r)
                col = col * d
                for j in range(3):
                    groundColors[3 * q + j] = col

# ------------------------------

glutInit()
initScene()

x = wp.vec3(0.0, 1.0, 2.0)
y = wp.vec3(1.0, -3.0, 0.0)
z = wp.sub(x, y)
print(str(z[0]) + "," + str(z[1]) + "," + str(z[2]))

glutInitDisplayMode(GLUT_RGBA)
glutInitWindowSize(800, 500)
glutInitWindowPosition(10, 10)
wind = glutCreateWindow("Parallel cloth simulation")

setupOpenGL()

glutDisplayFunc(displayCallback)
glutMouseFunc(mouseButtonCallback)
glutMotionFunc(mouseMotionCallback)
glutMouseWheelFunc(mouseWheelCallback)
glutKeyboardFunc(handleKeyDown)
glutKeyboardUpFunc(handleKeyUp)
glutTimerFunc(math.floor(1000.0 / targetFps), timerCallback, 0)


glutMainLoop()
