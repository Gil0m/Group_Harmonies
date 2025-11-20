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
        self.sep_radius = 15.0
        self.align_radius = 80.0
        self.coh_radius = 40.0
        self.sep_weight = 10.0
        self.align_weight = 5.0
        self.coh_weight = 3.0
        self.dt = 0.5

    def limit(self, vec, maxval):
        mag = np.linalg.norm(vec)
        if mag > maxval and mag > 1e-8:
            return vec / mag * maxval
        return vec
    
    def step(self, external_actions=None):
    # external_actions: dict boid_id -> accel vector (for controlled boid)

        positions = np.array([boid.pos for boid in self.boids])
        velocities = np.array([boid.vel for boid in self.boids])

        new_positions = []
        new_velocities = []
        # print('boids debut step', self.boids)

        for i, b in enumerate(self.boids):
            # find neighbors
            rel = positions - b.pos
            dist = np.linalg.norm(rel, axis=1)

            # separation
            close_mask = (dist > 0) & (dist < self.sep_radius)
            sep = np.zeros(3)
            if np.any(close_mask):
                sep = -np.sum(rel[close_mask] / (dist[close_mask][:, None]**2 + 1e-6), axis=0)

            # alignment
            align_mask = (dist > 0) & (dist < self.align_radius)
            align = np.zeros(3)
            if np.any(align_mask):
                align = np.mean(velocities[align_mask], axis=0) - b.vel

            # cohesion
            coh_mask = (dist > 0) & (dist < self.coh_radius)
            coh = np.zeros(3)
            if np.any(coh_mask):
                center = np.mean(positions[coh_mask], axis=0)
                coh = center - b.pos

            # combine
            accel = (
                self.sep_weight * sep**2 +
                self.align_weight * align**2 +
                self.coh_weight * coh**2
            )

            # external control
            if external_actions is not None and b.id in external_actions:
                # print('ext_act_accel', external_actions[b.id])
                # print(accel)
                accel += external_actions[b.id]

            # limit acceleration
            accel = self.limit(accel, self.max_acc)
            # print('accel', accel)

            # update velocity and position
            new_vel = b.vel + accel * self.dt
            new_vel = self.limit(new_vel, self.max_speed)
            new_pos = b.pos + new_vel * self.dt
            # print('new pos', new_pos)
            # print('bounds', self.bounds)

            # world wrap-around
            for k in range(3):
                if new_pos[k] < -self.bounds[k]:
                    new_pos[k] += 2 * self.bounds[k]
                elif new_pos[k] > self.bounds[k]:
                    new_pos[k] -= 2 * self.bounds[k]

            new_positions.append(new_pos)
            new_velocities.append(new_vel)

            # update all boids after the loop (avoid in-place update bias)
        for i, b in enumerate(self.boids):
            b.pos = new_positions[i]
            b.vel = new_velocities[i]

                    
    def get_state(self):
        return np.array([b.pos for b in self.boids]), np.array([b.vel for b in self.boids])
    



