import numpy as np

class Boid:
    def __init__(self, position, velocity, id):
        self.pos = position
        self.vel = velocity
        self.id = id

class Flock:
    def __init__(self, n, bounds, max_speed = 2.0, max_acc = 0.5, seed = None):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.bounds = np.array(bounds, dtype=np.float32)
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.boids = []
        for i in range(n):
            pos = (self.rng.random(3) - 0.5) * 2 * self.bounds
            vel = (self.rng.random(3) - 0.5) * 2 * max_speed
            self.boids.append(Boid(pos, vel, id=i))

        # Parameters
        self.sep_radius = 10.0
        self.align_radius = 10.0
        self.coh_radius = 10.0
        self.sep_weight = 2.0
        self.align_weight = 2.0
        self.coh_weight = 2.0
        self.dt = 1.0

    def limit(self, vec, maxval):
        mag = np.linalg.norm(vec)
        if mag > maxval and mag > 1e-8:
            return vec / mag * maxval
        return vec

    def separationAccForOneBoid(self, boid, neighbors):
        if len(neighbors) == 0:
            return np.zeros(3)
        rel = neighbors - boid.pos
        dist = np.linalg.norm(rel, axis=1)
        close_mask = (dist > 0) & (dist < self.sep_radius)
        if np.any(close_mask):
            return -np.sum(rel[close_mask] / (dist[close_mask][:, None]**2 + 1e-6), axis=0)
        return np.zeros(3)
    
    def alignAccForOneBoid(self, boid, neighbors):
        if len(neighbors) == 0:
            return np.zeros(3)
        rel = neighbors - boid.pos
        dist = np.linalg.norm(rel, axis=1)
        align_mask = (dist > 0) & (dist < self.align_radius)
        if np.any(align_mask):
            return np.mean([b.vel for b in self.boids if align_mask[b.id]], axis=0) - boid.vel
        return np.zeros(3)
    
    def cohAccForOneBoid(self, boid, neighbors):
        if len(neighbors) == 0:
            return np.zeros(3)
        rel = neighbors - boid.pos
        dist = np.linalg.norm(rel, axis=1)
        coh_mask = (dist > 0) & (dist < self.coh_radius)
        if np.any(coh_mask):
            center = np.mean(rel[coh_mask], axis=0) + boid.pos
            return center - boid.pos
        return np.zeros(3)
    
    def computeAccCombinedSepAlignCoh(self, boid, neighbors):
        sep = self.separationAccForOneBoid(boid, neighbors)
        align = self.alignAccForOneBoid(boid, neighbors)
        coh = self.cohAccForOneBoid(boid, neighbors)
        return self.sep_weight * sep + self.align_weight * align + self.coh_weight * coh

    def checkWorldWrapAround(self, pos):
        for k in range(3):
            if pos[k] < -self.bounds[k]:
                pos[k] += 2 * self.bounds[k]
            elif pos[k] > self.bounds[k]:
                pos[k] -= 2 * self.bounds[k]
        return pos
    

    def step(self, external_actions=None):
    # external_actions: dict boid_id -> accel vector (for controlled boid)

        positions = np.array([boid.pos for boid in self.boids])
        velocities = np.array([boid.vel for boid in self.boids])

        new_positions = []
        new_velocities = []
        # print('boids debut step', self.boids)

        for i, b in enumerate(self.boids):

            accel = self.computeAccCombinedSepAlignCoh(b, positions)
            
            if external_actions is not None and b.id in external_actions:

                accel += external_actions[b.id]

            accel = self.limit(accel, self.max_acc)

            new_vel = 1 * b.vel + accel * self.dt
            new_vel = self.limit(new_vel, self.max_speed)
            new_pos = b.pos + new_vel * self.dt

            new_pos = self.checkWorldWrapAround(new_pos)

            new_positions.append(new_pos)
            new_velocities.append(new_vel)

            # update all boids after the loop (avoid in-place update bias)
        for i, b in enumerate(self.boids):
            b.pos = new_positions[i]
            b.vel = new_velocities[i]

                                        
    def get_state(self):
        return np.array([b.pos for b in self.boids]), np.array([b.vel for b in self.boids])
    

def leaderAccelerationPerpetualNoisy(t, pos, vel, max_acc):
    """
    Perpetual Movement for the leader of the flock.
    """
    # direction principale
    base_dir = np.array([
        np.cos(0.2 * t),
        np.sin(0.2 * t),
        0.5 * np.sin(0.1 * t)
    ])

    # bruit doux
    noise = 0.3 * np.array([
        np.sin(1.3 * t + 1.0),
        np.sin(1.7 * t + 2.0),
        np.sin(1.1 * t + 3.0)
    ])

    desired_direction = base_dir + noise

    # normalisation
    norm = np.linalg.norm(desired_direction)
    if norm > 1e-6:
        desired_direction = desired_direction / norm * max_acc

    return desired_direction
