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

wp.init()

# ---------------------------------------------

targetFps = 80
numSubsteps = 30
timeStep = 1.0 / 80.0
# timeStep = 3.0
gravity = wp.vec3(0.0, -9.0, 0.0)
velMax = 0.03
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

class Hash:
    def __init__(self, spacing, maxNumObjects):
        self.spacing = spacing
        self.tableSize = 2 * maxNumObjects
        self.cellStart = [0] * (self.tableSize + 1)
        self.cellEntries = [0] * maxNumObjects
        self.queryIds = [0] * (2 * maxNumObjects)
        self.querySize = 0

    def hashCoords(self, xi, yi, zi):
        h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)  # fantasy function
        return abs(h) % self.tableSize

    def intCoord(self, coord):
        return int(coord / self.spacing)

    def hashPos(self, pos, nr):
        return self.hashCoords(
            self.intCoord(pos[nr][0]),
            self.intCoord(pos[nr][1]),
            self.intCoord(pos[nr][2])
        )

    def create(self, pos):
        numObjects = min(len(pos), len(self.cellEntries))

        # determine cell sizes
        self.cellStart = [0] * len(self.cellStart)
        self.cellEntries = [0] * len(self.cellEntries)

        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] += 1

        # determine cells starts
        start = 0
        for i in range(self.tableSize):
            start += self.cellStart[i]
            self.cellStart[i] = start
        self.cellStart[self.tableSize] = start  # guard

        # fill in objects ids
        for i in range(numObjects):
            h = self.hashPos(pos, i)
            self.cellStart[h] -= 1
            self.cellEntries[self.cellStart[h]] = i

    def query(self, pos, np, nq, nr, maxDist):

        xMin = min(pos[np][0], pos[nq][0], pos[nr][0])
        xMax = max(pos[np][0], pos[nq][0], pos[nr][0])
        yMin = min(pos[np][1], pos[nq][1], pos[nr][1])
        yMax = max(pos[np][1], pos[nq][1], pos[nr][1])
        zMin = min(pos[np][2], pos[nq][2], pos[nr][2])
        zMax = max(pos[np][2], pos[nq][2], pos[nr][2])

        x0 = self.intCoord(xMin - maxDist)
        y0 = self.intCoord(yMin - maxDist)
        z0 = self.intCoord(zMin - maxDist)

        x1 = self.intCoord(xMax + maxDist)
        y1 = self.intCoord(yMax + maxDist)
        z1 = self.intCoord(zMax + maxDist)

        self.querySize = 0

        for xi in range(x0, x1 + 1):
            for yi in range(y0, y1 + 1):
                for zi in range(z0, z1 + 1):
                    h = self.hashCoords(xi, yi, zi)
                    start = self.cellStart[h]
                    end = self.cellStart[h + 1]

                    for i in range(start, end):
                        if self.querySize >= 6890:
                            print("QS =", self.querySize)
                        self.queryIds[self.querySize] = self.cellEntries[i]
                        self.querySize += 1
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
        bodyVertices, bodyTriangles = self.readOBJFile(bodyOBJFilePath)
        self.numParticles = len(vertices)
        self.numBodyParticles = len(bodyVertices)
        pos = np.zeros(3 * self.numParticles)
        bodyPos = np.zeros(3 * len(bodyVertices))
        normals = np.zeros(3 * self.numParticles)
        invMass = np.zeros(self.numParticles)

        mnY = 0
        for v in bodyVertices:
            mnY = min(mnY, v[1])
        for i in range(self.numParticles):
            pos[3 * i] = vertices[i][0]
            pos[3 * i + 1] = vertices[i][1] - mnY 
            pos[3 * i + 2] = vertices[i][2]
            invMass[i] = 1

        for i in range(len(bodyVertices)):
            bodyPos[3 * i] = bodyVertices[i][0]
            bodyPos[3 * i + 1] = bodyVertices[i][1] - mnY
            bodyPos[3 * i + 2] = bodyVertices[i][2]

        spacing = 0.001

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

        wp.launch(kernel = self.computeRestLengths,
                inputs = [self.pos, self.distConstIds, self.constRestLengths], 
                dim = self.numDistConstraints,  device = "cuda")

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

        self.triDist = wp.zeros(self.numTris, dtype = float, device = "cuda")
        self.hostTriDist = wp.zeros(self.numTris, dtype = float, device = "cpu")

        maxDist = 0.0008
        self.hash = Hash(spacing, len(bodyVertices))
        self.hash.create(self.hostBodyPos.numpy())
        start_time = time.time()
        self.numCollisions = self.queryAllCloseBodyParticles(self.hostTriIds, self.hostPos.numpy(), maxDist)
        print("Num collisions =", self.numCollisions)
        end_time = time.time()
        print("Elapsed time:", end_time - start_time, "seconds")
        print(str(self.numTris) + " triangles created")
        print(str(self.numDistConstraints) + " distance constraints created")
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
    def collide_cloth_tri_body_particle(
        numParticles: int,
        pos: wp.array(dtype = wp.vec3),
        bodyPos: wp.array(dtype = wp.vec3),
        clothTriIds: wp.array(dtype = wp.int32),
        f: wp.array(dtype = wp.vec3)
    ):
        tid = wp.tid()
        triNo = tid // numParticles
        particleNo = tid % numParticles

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
        fn = n * c * 3e5
        wp.atomic_add(f, i, fn * bary[0])
        wp.atomic_add(f, j, fn * bary[1])
        wp.atomic_add(f, k, fn * bary[2])

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
        fn = n * c * 3e5
        wp.atomic_add(f, i, fn * bary[0])
        wp.atomic_add(f, j, fn * bary[1])
        wp.atomic_add(f, k, fn * bary[2])
        

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
    def queryAllCloseBodyParticles(self, triIds, pos, maxDist):
        num = 0
        numTris = len(triIds) // 3
        for iTri in range(numTris):
            self.hash.query(pos, triIds[3 * iTri], triIds[3 * iTri + 1], triIds[3 * iTri + 2], maxDist)
            if iTri == 12000:
                self.renderParticles1.append(triIds[3 * iTri])
                self.renderParticles1.append(triIds[3 * iTri + 1])
                self.renderParticles1.append(triIds[3 * iTri + 2])
            for j in range(self.hash.querySize):
                if num >= len(self.adjIds):
                    newIds = [0] * (2 * num)  # dynamic array
                    newIds[:num] = self.adjIds
                    self.adjIds = newIds

                    newIds1 = [0] * (2 * num)
                    newIds1[:num] = self.adjIdsTri
                    self.adjIdsTri = newIds1
                self.adjIds[num] = self.hash.queryIds[j]
                self.adjIdsTri[num] = iTri 
                if iTri == 12000:
                    self.renderParticles2.append(self.adjIds[num])
                num += 1
        self.adjIdsCuda = wp.array(self.adjIds[:num], dtype = wp.int32, device = "cuda")
        self.adjIdsTriCuda = wp.array(self.adjIdsTri[:num], dtype = wp.int32, device = "cuda")
        return num

    # ----------------------------------
    def simulate(self):

        # ----------------------------------
        dt = timeStep / numSubsteps
        numPasses = len(self.passSizes)

        start_time = time.time()
        for step in range(numSubsteps):  
            # wp.launch(kernel = self.integrate, 
            #     inputs = [dt, gravity, self.invMass, self.prevPos, self.pos, self.vel, self.sphereCenter, self.sphereRadius], 
            #     dim = self.numParticles, device = "cuda")
            wp.launch(kernel = self.integrate_gravity, 
                inputs = [dt, velMax, gravity, self.invMass, self.prevPos, self.pos, self.vel], 
                dim = self.numParticles, device = "cuda")

            if solveType == 0:
                firstConstraint = 0
                for passNr in range(numPasses):
                    numConstraints = self.passSizes[passNr]

                    if self.passIndependent[passNr]:
                        wp.launch(kernel = self.solveDistanceConstraints,
                            inputs = [0, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                            dim = numConstraints,  device = "cuda")
                    else:
                        self.corr.zero_()
                        wp.launch(kernel = self.solveDistanceConstraints,
                            inputs = [1, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                            dim = numConstraints,  device = "cuda")
                        wp.launch(kernel = self.addCorrections,
                            inputs = [self.pos, self.corr, jacobiScale], 
                            dim = self.numParticles,  device = "cuda")
                    
                    firstConstraint = firstConstraint + numConstraints

            elif solveType == 1:
                self.corr.zero_()
                wp.launch(kernel = self.solveDistanceConstraints, 
                    inputs = [1, 0, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths], 
                    dim = self.numDistConstraints,  device = "cuda")
                wp.launch(kernel = self.addCorrections,
                    inputs = [self.pos, self.corr, jacobiScale], 
                    dim = self.numParticles,  device = "cuda")

            self.corr.zero_()
            # wp.launch(kernel = self.collide_cloth_tri_body_particle,
            #           inputs = [self.numBodyParticles, self.pos, self.bodyPos, 
            #                     self.triIds, self.corr],
            #           dim = self.numTris * self.numBodyParticles,
            #           device = "cuda")
            wp.launch(kernel = self.collide_cloth_tri_body_particle_fast,
                      inputs = [self.adjIdsCuda, self.adjIdsTriCuda, self.pos, self.bodyPos, 
                                self.triIds, self.corr],
                      dim = self.numCollisions,
                      device = "cuda")

            wp.launch(kernel = self.integrate_body_collisions, 
                      inputs = [dt, self.invMass, self.pos, self.corr], 
                      dim = self.numParticles, device = "cuda")

            wp.launch(kernel = self.updateVel, 
                inputs = [dt, self.prevPos, self.pos, self.vel], dim = self.numParticles, device = "cuda")
            
        wp.copy(self.hostPos, self.pos)
        end_time = time.time()
        print("Elapsed time 1:", end_time - start_time, "seconds")

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
            print(id)
            glPushMatrix()
            p = pos[id]
            print(p[0], p[1], p[2])
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

    glColor3f(1.0, 1.0, 1.0)
    glNormal3f(0.0, 1.0, 0.0)

    numVerts = math.floor(len(groundVerts) / 3)

    glVertexPointer(3, GL_FLOAT, 0, groundVerts)
    glColorPointer(3, GL_FLOAT, 0, groundColors)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glDrawArrays(GL_QUADS, 0, numVerts)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

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
