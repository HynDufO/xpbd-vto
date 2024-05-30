# STAGE 2: FROM 3D HUMAN MODEL TO CLOTH TRY-ON
## Set up 
1. Install necessary packages (recommend using `conda`)
    - Numpy: `pip install numpy`
    - Warp: `pip install warp-lang`
    - PyOpenGl: `pip install PyOpenGL PyOpenGL_accelerate`
2. Generate necessary files for the cloth
    - If using XPBD:
        - This involves running: `cloth_body_constraints.cpp`, `graph_coloring.py`, `query_close_points.cpp`. (for `.cpp` just run normally with `g++ -O2 cloth_body_constraints.cpp -o C && ./C` or something like this).
        - The names of the output files of those above are hardcoded in the corresponding `gpu*.py` file.
    - If using LBS:
        - This involves running: `get_corr_weights.cpp`.
3. Run the version you want (`gpu*.py`)
    ```
    python mock_pose_estimator.py | python gpu.py
    ```
## Structure
`cloth_body_constraints.cpp`:
- From the body and cloth obj files, with some hardcoded vertices of the cloth, find the nearest vertices on the body and create a distance constraint between the cloth and the body 
- To improve stability so that the cloth does not fly off the body.
- Output to `cloth_body_constraints.txt`

`get_corr_weights.cpp`:
- For each vertex of the cloth, find the nearest vertex on the body -> To copy weights for LBS
- Output to `corr_weights*.txt`

`graph_coloring.py`:
- Greedy algorithm to group constraints used in Gauss-Seidel and Jacobi.
- `graph_coloring_exclude_lbs.py`: If both vertices of a constraint are within the vertices using LBS, then it is redundant and there's no need to use XBPD for this constraint.
- Output to `*_constraints_groups.txt`

`query_close_points.cpp`:
- No need to compute collision for vertex-triangle pairs that are guaranteed to never collide. 
- For a cloth vertex, just find the 20-30 closest triangles on the body and vice versa for a body vertex and cloth triangles.
- There is a custom condition: if a vertex is in an area prone to multiple collisions, the number of triangles will increase...
- Output to `*tri_point_pairs*.txt`

`mock_pose_estimator.py`:
- Read from a pose file (`.npy`) and output `24x3` numbers for each frame.

`gpu*.py`: Main file connecting everything:
- `gpu.py`: Uses XBPD.
- `gpu_lbs`: Uses LBS.
- `gpu_combine`: Uses both XPBD and LBS (LBS for vertices on the sleeves and XPBD for the remaining vertices).
