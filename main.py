# Pytorch
import torch
import torch.nn.functional as F
# For maths constants
import math
# To debug AD
#from torchviz import make_dot
# For command parsing
from optparse import OptionParser

# Display results
DISPLAY = True
# If we want to save Absolute gradient values
ABS_GRAD = True
SAVE_EXR = True
# Rendering constant
WIDTH = 1024
HEIGHT = 768
OPENING = .5135
PRECISION = torch.float64
# 1 = direct
NB_BOUNCE = 1
SPP = 1
CAPTURE_FIREFLIES = False

# What we want to compute
COMPUTE_FINITE = True
COMPUTE_AD = True

# If we want to use GPU or not
USE_GPU = False
if USE_GPU:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class ObjDrawing:
    """Wavefront line exporter. For debugging proposes"""
    def __init__(self):
        super().__init__()
        self.lines = {}
        self.current = None
        self.capture = False
    
    def group(self, name):
        self.current = name
        if not(name in self.lines):
            self.lines[name] = []

    def line(self, x0, y0, z0, x1, y1, z1):
        if self.capture and self.current != None:
            self.lines[self.current] += [(x0, y0, z0), (x1, y1, z1)]

    def write(self, output):
        if len(self.lines) == 0:
            print(f"WARN: No lines have been capture ({output}.obj)")
            return
        print(f"Export {len(self.lines)} lines in OBJ ({output}.obj)...")
        f = open(f"{output}.obj", "w")
        offset = 0
        for o in self.lines.keys():
            f.write(f"o {o}\n")
            for l in self.lines[o]:
                f.write(f"v {l[0]} {l[1]} {l[2]}\n")
            for i in range(0, len(self.lines[o]), 2):
                f.write(f"l {i+1+offset} {i+2+offset}\n")
            offset += len(self.lines[o])
        

# Helpers
def imshow(ax, t, shape):
    """Function to imshow a buffer"""
    return ax.imshow(t.reshape(shape))

def dot(t1, t2):
    """Dot product between two vector."""
    return torch.mul(t1, t2).sum(axis=1)

class Sphere:
    def __init__(self, radius, center, color, emission=[0.0, 0.0, 0.0]):
        self.center = torch.tensor(center, dtype=PRECISION, device=DEVICE)
        self.radius = torch.tensor(radius, dtype=PRECISION, device=DEVICE)
        self.color = torch.tensor(color, dtype=PRECISION, device=DEVICE)
        self.emission = torch.tensor(emission, dtype=PRECISION, device=DEVICE)

    def intersect(self, ray_org, ray_dir):
        """Do the ray intersection. If the intersection is not possible return 1e5"""
        num = ray_org.shape[0]
        miss = torch.ones(num, dtype=PRECISION, device=DEVICE) * 1e5
        EPS = 1e-5
        op = (self.center - ray_org)
        b = dot(op, ray_dir)
        det = b * b - dot(op, op) + self.radius * self.radius
        det2 = torch.sqrt(det.relu() + 1e-8)  # det.where(det <= 0, det.sqrt())
        t = b - det2
        t2 = b + det2
        return miss.where(det <= 0.0, t.where(t > EPS, t2.where(t2 > EPS, miss)))


def intersect(pos, dir):
    """Naive procedure that intersect a scene"""
    distances = [s.intersect(pos, dir) for s in scene]
    min_distance, idx = torch.min(torch.stack(distances), dim=0)
    
    return min_distance, idx, min_distance < 1e5

def unit_sphere_sampling(u0, u1):
    """Sample an unit sphere"""
    theta  = u0 * 2 * math.pi
    phi = u1 * math.pi 
    sphi = phi.sin()
    return torch.stack([theta.cos() * sphi, theta.sin() * sphi, phi.cos()], dim=1)

def generate_primary_rays(cam_pos, cam_dir, u):
    """Generate ray from the sensor"""
    x, y = torch.meshgrid(torch.arange(0, HEIGHT), torch.arange(0, WIDTH))
    x = x.flatten()
    y = y.flatten()
    if USE_GPU:
        x = x.cuda()
        y = y.cuda()
    x = HEIGHT - x - 1

    # This code do not work...
    # Hard coded values below
    #torch.fmod(cx, cam_dir)
    # cy[0] = 0.0 # fmod (0.0)
    #cy = (cy / torch.sqrt(torch.dot(cy, cy))) * OPENING
    cx = torch.tensor([WIDTH * OPENING / HEIGHT, 0, 0], dtype=PRECISION, device=DEVICE)
    cy = torch.tensor([0, 0.513034, -0.0218614], dtype=PRECISION, device=DEVICE)

    # Generate positon and direction as smallPT does
    d = ((u[:, 0] + x) / HEIGHT - .5).reshape(x.shape[0],1).type(dtype=PRECISION).matmul(cy.reshape(1, 3))
    d += ((u[:, 1] + y) / WIDTH - .5).reshape(y.shape[0], 1).type(dtype=PRECISION).matmul(cx.reshape(1, 3))
    d += cam_dir
    cam_pos = cam_pos + d * 140
    d = F.normalize(d, dim=1, p=2)

    return cam_pos, d

def rendering_sensor(cam_pos, cam_dir, u, obj, active=None):
    pos, dir = generate_primary_rays(cam_pos, cam_dir, u)

    # Create buffers
    # - The accumulation buffer
    color = torch.zeros((dir.shape[0], 3), dtype=PRECISION, device=DEVICE)
    # - The path throughput
    throughput = torch.ones((dir.shape[0], 3), dtype=PRECISION, device=DEVICE)
    # - Mark which ray are actives
    if active == None:
        active = torch.ones(pos.shape[0], dtype=torch.bool, device=DEVICE)
    # - If the ray is emitted from a dirac interaction
    dirac = torch.ones(pos.shape[0], dtype=torch.bool, device=DEVICE)
    return rendering(pos, dir, color, throughput, u, NB_BOUNCE, 2, active, dirac, obj)

def rendering(pos, dir, color, throughput, u, depth, u_index, active, dirac, obj):
    # Intersection
    its, idx, hit = intersect(pos, dir)

    if obj.capture:
        with torch.no_grad():
            hit = hit & active
            obj.group(f"hit_{depth}")
            hit_ori_pos = pos[hit]
            hit_next_pos = pos[hit] + dir[hit] * its[hit].unsqueeze(1)
            for i in range(hit_ori_pos.shape[0]):
                obj.line(hit_ori_pos[i, 0], hit_ori_pos[i, 1], hit_ori_pos[i, 2], 
                hit_next_pos[i, 0], hit_next_pos[i, 1], hit_next_pos[i, 2])

    # Create all the input for the next call
    # Maybe some are not necessary but it is important
    # for autodiff to avoid inplace operations
    active_next = hit & active
    dirac_next = torch.zeros_like(dirac)
    pos_next = torch.zeros_like(pos)
    dir_next = torch.zeros_like(dir)
    throughput_next = torch.zeros_like(throughput)

    for (i, s) in enumerate(scene):
        # Create the mask for hitting this particular object
        hit_object = active_next & (idx == i)
        if hit_object.any():
            # We retrive the self emission if we are doing primitive path tracing
            # or we arrive from a impulse BSDF
            color[hit_object] += dirac[hit_object].unsqueeze(1) * throughput[hit_object] * s.emission
            
            # Extract all buffers
            p_select = pos[hit_object]
            d_select = dir[hit_object]
            dist_select = its[hit_object]

            # Compute the intersection point
            p_its = p_select + d_select * dist_select.unsqueeze(1)

            # Compute normal (from the sphere)
            n = p_its - s.center
            n = F.normalize(n, dim=1, p=2)
            # And make sure that the normal is on the right direction 
            dot_n_view_dir = -dot(n, d_select)
            flip = 1 - 2 * dot_n_view_dir.lt(0).double()
            n_correct = n * flip.unsqueeze(1)
            
            #########################
            # Explicit light connection
            #########################
            # Sample a point on the light
            u_select_0 = u[hit_object, u_index]
            u_select_1 = u[hit_object, u_index + 1]
            n_light = unit_sphere_sampling(u_select_0, u_select_1)
            p_light = scene[-1].center + n_light * scene[-1].radius

            # Write the object
            if obj.capture:
                with torch.no_grad():
                    obj.group(f"explicit_{depth}")
                    hit_ori_pos = p_its
                    hit_next_pos = p_light 
                    for i in range(hit_ori_pos.shape[0]):
                        obj.line(hit_ori_pos[i, 0], hit_ori_pos[i, 1], hit_ori_pos[i, 2], 
                        hit_next_pos[i, 0], hit_next_pos[i, 1], hit_next_pos[i, 2])

            # Do connection (without checking visibility)
            d_light = p_light - p_its
            d_length_sq = dot(d_light, d_light)
            d_light_norm = F.normalize(d_light, dim=1, p=2)

            # Compute geometry factor
            geom = dot(d_light_norm, n_correct).clamp(0, 1)
            geom *= dot(-d_light_norm, n_light).clamp(0, 1)
            geom /= d_length_sq
            #geom = 1.0 / d_length_sq

            # Compute visibility
            its_light, idx_light, hit_light = intersect(p_its, d_light_norm)
            visible = hit_light & (idx_light == len(scene) - 1) # This is correct for the sphere light
            geom *= visible.double()
            

            if obj.capture:
                print("dot surf :", dot(d_light, n_correct))
                print("dot light:", dot(-d_light, n_light))
                print("inv dist :", 1.0 / d_length_sq)
                print("visibile :", visible)

            # Compute shading
            factor = 4 * (scene[-1].radius**2) # Pi factor get cancel out
            color[hit_object] += scene[-1].emission * s.color * geom.unsqueeze(1) * factor

            # Code to continue to trace the ray
            # Using BSDF sampling
            if depth > 1:
                # Random numbers (to sample the outgoing direction)
                u_select_0 = u[hit_object, u_index + 2]
                u_select_1 = u[hit_object, u_index + 3]
        
                # Sample hemisphere
                dir_diffuse = unit_sphere_sampling(u_select_0, u_select_1)

                # Flip the vector that produce wrong direction
                dot_n_dir = dot(n_correct, dir_diffuse)
                flip = 1 - 2 * dot_n_dir.lt(0).double()
                dir_diffuse_flip = dir_diffuse * flip.unsqueeze(1)

                # Update the throughput (diffuse)
                factor = dot_n_dir.abs() 
                throughput_next[hit_object] = throughput[hit_object] * factor.unsqueeze(1) * s.color
                
                # Write back the new direction & Position
                dir_next[hit_object] = dir_diffuse_flip
                pos_next[hit_object] = p_its

    # We stop at depth 1 as we do explicit light connection
    if depth == 1:
        return color
    else:
        return rendering(pos_next, dir_next, color, throughput_next, u, depth - 1, u_index + 4, active_next, dirac_next, obj)


if __name__ == "__main__":
    # Get arugments
    parser = OptionParser()
    parser.add_option("-s", dest="spp",
                  help="number of SPP", default="1", type="int")
    parser.add_option("-b", dest="bounce",
                  help="number of bounce", default="1", type="int")
    parser.add_option("-i", dest="img_size",
                  help="image size (widthxheight)", default="300x300")
    parser.add_option("-o", dest="output",
                  help="output EXR filename", default="")
    parser.add_option("--autodiff", dest="autodiff",
                  help="compute derivatives with auto diff", default=False, action="store_true")
    parser.add_option("--finitediff", dest="finitediff",
                  help="compute derivatives with finite diff", default=False, action="store_true")
    parser.add_option("-d", dest="display", 
                  help="shows gradient with pyplot", default=False, action="store_true")
    parser.add_option("--release", dest="release", 
                  help="remove some debugging operations", default=False, action="store_true")
    
    (options, args) = parser.parse_args()

    # Fill the arguments
    SPP = options.spp
    NB_BOUNCE = options.bounce
    COMPUTE_AD = options.autodiff
    COMPUTE_FINITE = options.finitediff
    if options.output == "":
        SAVE_EXR = False
    else:
        SAVE_EXR = True
        OUTPUT = options.output
    size = [int(v) for v in options.img_size.split("x")]
    print("Image size:", size)
    WIDTH = size[0]
    HEIGHT = size[1]
    DISPLAY = options.display

    torch.manual_seed(0) # Make the seed deterministic (easier debug)
    if not(options.release):
        # Might slow down quite big the program...
        # But very useful in practice (for development)
        torch.autograd.set_detect_anomaly(True)
        # cudnn and other options (certainly optional)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For debugging
    obj = ObjDrawing()

    scene = [
        Sphere(1e5, [1e5 + 1, 40.8, 81.], [.75, .25, .25]),
        Sphere(1e5, [-1e5 + 99, 40.8, 81.6], [.25, .25, .75]),
        Sphere(1e5, [50, 40.8, 1e5], [.75, .75, .75]),
        Sphere(1e5, [50, 40.8, -1e5 + 170], [0.0, 0.0, 0.0]),
        Sphere(1e5, [50, 1e5, 81.6], [.75, .75, .75]),
        Sphere(1e5, [50, -1e5 + 81.6, 81.6], [.75, .75, .75]),
        Sphere(16.5, [27, 16.5, 47], [.999, .999, .999]),
        Sphere(16.5, [73, 16.5, 78], [.999, .999, .999]),
        # Larger light if doing implicit light interesection (smallpt by default)
        #Sphere(600, [50, 681.6 - .27, 81.6], [0.0, 0.0, 0.0]),
        # Smaller light for explicit light connection
        Sphere(1.5, [50,81.6-16.5,81.6], [0.0, 0.0, 0.0]),
    ]
    # The last sphere is the light source
    # scene[-1].emission = torch.tensor([12,12,12], dtype=PRECISION)
    scene[-1].emission = torch.tensor([400,400,400], dtype=PRECISION, device=DEVICE)

    # Create camera
    cam_pos = torch.tensor([50, 52, 295.6], dtype=PRECISION, device=DEVICE)
    cam_dir = torch.tensor([0, -0.042612, -1], dtype=PRECISION, device=DEVICE)
    cam_dir = cam_dir / torch.sqrt(torch.dot(cam_dir, cam_dir))

    # The dimension of the random number vector
    DIM_U = 2 + 4 * (NB_BOUNCE - 1) + 2
    
    # Variables for the autodiff 
    color = torch.zeros((WIDTH*HEIGHT, 3), dtype=PRECISION, device=DEVICE)
    uGradR = torch.zeros((WIDTH*HEIGHT, DIM_U), dtype=PRECISION, device=DEVICE)
    uGradG = torch.zeros((WIDTH*HEIGHT, DIM_U), dtype=PRECISION, device=DEVICE)
    uGradB = torch.zeros((WIDTH*HEIGHT, DIM_U), dtype=PRECISION, device=DEVICE)
    gradFiniteAll = []
    if COMPUTE_FINITE:
        # For finite difference, the gradient are stored differently 
        for _ in range(DIM_U):
            gradFiniteAll.append(torch.zeros((WIDTH*HEIGHT, 3), dtype=PRECISION, device=DEVICE))

    # Do the rendering 
    for nsample in range(SPP):
        print(f"Rendering ... {nsample+1}/{SPP}")
        u = torch.rand((WIDTH * HEIGHT, DIM_U), requires_grad=COMPUTE_AD, device=DEVICE)
        obj.capture = False
        color_current = rendering_sensor(cam_pos, cam_dir, u, obj)

        if COMPUTE_AD:
            # DEBUG: Output the graph (useful to debug)
            #print("Debug: Writing AD graph...")
            #make_dot(color).render("attached", format="png")

            print(f"Compute backward... {nsample+1}/{SPP}")
            # Compute R,G,B derivatives
            # - R
            dervInput =  torch.zeros((WIDTH*HEIGHT, 3), device=DEVICE)
            dervInput[:, 0] = 1.0 
            color_current.backward(dervInput, retain_graph=True)
            uGradR += u.grad.clone().detach()
            u.grad.zero_()
            dervInput.zero_()
            # - G
            dervInput[:, 1] = 1.0
            color_current.backward(dervInput, retain_graph=True)
            uGradG += u.grad.clone().detach()
            u.grad.zero_()
            dervInput.zero_()
            # - B
            dervInput[:, 2] = 1.0
            color_current.backward(dervInput)
            uGradB += u.grad.clone().detach()
            u.grad.zero_()
        
        color += color_current

        
        if CAPTURE_FIREFLIES:
            with torch.no_grad():
                print("Capture fireflies...")
                obj.capture = True
                fireflies = uGradR[:, 2].abs() > 10.0
                fireflies = uGradG[:, 2].abs() > 10.0
                fireflies = uGradB[:, 2].abs() > 10.0
                print(f"Number of fireflies {fireflies.sum()}")
                print("Colors", color[fireflies])
                print("GradR", uGradR[fireflies])
                print("GradG", uGradR[fireflies])
                print("GradB", uGradR[fireflies])
                #print(fireflies)
                rendering_sensor(cam_pos, cam_dir, u, obj, fireflies)
                obj.capture = False

        if COMPUTE_FINITE:
            print(f"Compute finite difference ... {nsample+1}/{SPP}")
            # Use finite difference
            DELTA = .5 # For the image space
            DELTA2 = 0.01
            with torch.no_grad():
                # no_grad is necessary as `u`
                # have gradient representation
                for i in range(DIM_U):
                    # Make a copy of random numbers
                    # Note that as u
                    u_new_fwd = u.clone()
                    u_new_bak = u_new_fwd.clone()

                    if i == 0 or i == 1:
                        # We need to use bigger delta for the image space
                        # gradient, as the random number have less influence here
                        u_new_fwd[:, i] += DELTA / 2
                        u_new_bak[:, i] -= DELTA / 2
                    else:
                        u_new_fwd[:, i] += DELTA2 / 2
                        u_new_bak[:, i] -= DELTA2 / 2

                    u_new_fwd = u_new_fwd.clamp(0.0, 1.0)
                    u_new_bak = u_new_bak.clamp(0.0, 1.0)
                    delta = u_new_fwd[:, i] - u_new_bak[:, i]

                    # Central diff
                    diffFwd = rendering_sensor(cam_pos, cam_dir, u_new_fwd, obj)
                    diffBak = rendering_sensor(cam_pos, cam_dir, u_new_bak, obj)
                    gradFinite = (diffFwd - diffBak)
                    
                    # Normalize and accumulate the result
                    gradFinite /= delta.unsqueeze(1)
                    gradFiniteAll[i] += gradFinite

    with torch.no_grad():
        # Normalizing with the number of samples
        color /= SPP
        uGradR /= SPP
        uGradG /= SPP 
        uGradB /= SPP
        if COMPUTE_FINITE:
            for i in range(DIM_U):
                gradFiniteAll[i] /= SPP

        # In case of GPU we need to transfer everything back to the CPU
        if USE_GPU:
            color = color.cpu()
            uGradR = uGradR.cpu()
            uGradG = uGradG.cpu()
            uGradB = uGradB.cpu()
            if COMPUTE_FINITE:
                for i in range(DIM_U):
                    gradFiniteAll[i] = gradFiniteAll[i].cpu()

        # Make AD computed gradient the same structure
        # of the finite difference gradient
        gradAll = [torch.stack([uGradR[:, i], uGradG[:, i], uGradB[:, i]], dim=1) for i in range(u.shape[1])]

        if SAVE_EXR:
            import pyexr
            print("Write EXR...")
            pyexr.write(f"{OUTPUT}_color.exr", color.reshape((HEIGHT,WIDTH, 3)).type(dtype=torch.float32).numpy())
            if COMPUTE_AD:
                for i, g in enumerate(gradAll):
                    value = gradAll[i].reshape((HEIGHT,WIDTH, 3)).type(dtype=torch.float32)
                    if ABS_GRAD:
                        value = value.abs()
                    pyexr.write(f"{OUTPUT}_AD_{i}.exr", value.numpy())
            if COMPUTE_FINITE:
                for i, g in enumerate(gradFiniteAll):
                    value = gradFiniteAll[i].reshape((HEIGHT,WIDTH, 3)).type(dtype=torch.float32)
                    if ABS_GRAD:
                        value = value.abs()
                    pyexr.write(f"{OUTPUT}_finite_{i}.exr", value.numpy())

        obj.write(OUTPUT)

        if DISPLAY:
            import matplotlib.pyplot as plt
            print("Showing...")
            fig_trace, axis = plt.subplots(1, 1, figsize=(15, 7))
            imshow(axis, color.clamp(0.0, 1.0), (HEIGHT,WIDTH, 3))

            # Function to tonemap gradients
            def tonemap(buffer):
                return buffer.abs().clamp(0.0, 1.0) #/ 0.1 

            # Compute the grid for outputing the computed gradients
            nbW = int(math.sqrt(u.shape[1]))
            nbH = u.shape[1] // nbW

            # Show AD gradients
            if COMPUTE_AD:
                fig_auto, axis = plt.subplots(nbW, nbH, figsize=(15, 7))
                for i, (ax, arr) in enumerate(zip(axis.flatten(), gradAll)):
                    pcm = imshow(ax, tonemap(arr), (HEIGHT,WIDTH, 3))

            # If we compute finite difference
            if COMPUTE_FINITE:
                fig_auto, axis = plt.subplots(nbW, nbH, figsize=(15, 7))
                for i, (ax, arr) in enumerate(zip(axis.flatten(), gradFiniteAll)):
                    pcm = imshow(ax, tonemap(arr), (HEIGHT,WIDTH, 3))

            plt.show()
