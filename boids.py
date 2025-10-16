# boids.py


# parameters
self.sep_radius = 1.0
self.align_radius = 3.0
self.coh_radius = 3.0
self.sep_weight = 1.5
self.align_weight = 1.0
self.coh_weight = 1.0
self.dt = 0.1


def limit(self, vec, maxval):
mag = np.linalg.norm(vec)
if mag > maxval and mag > 1e-8:
return vec / mag * maxval
return vec


def step(self, external_actions=None):
# external_actions: dict boid_id -> accel vector (for controlled boid)
positions = np.array([b.pos for b in self.boids])
velocities = np.array([b.vel for b in self.boids])
for i, b in enumerate(self.boids):
# find neighbors
rel = positions - b.pos
dist = np.linalg.norm(rel, axis=1)


# separation
close_mask = (dist > 0) & (dist < self.sep_radius)
sep = np.zeros(3)
if np.any(close_mask):
# push away proportionally
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


accel = (self.sep_weight * sep + self.align_weight * align + self.coh_weight * coh)


# add external action if present
if external_actions is not None and b.id in external_actions:
accel += external_actions[b.id]


accel = self.limit(accel, self.max_acc)
b.vel = b.vel + accel * self.dt
b.vel = self.limit(b.vel, self.max_speed)
b.pos = b.pos + b.vel * self.dt


# world bounds: wrap-around
for k in range(3):
if b.pos[k] < -self.bounds[k]:
b.pos[k] += 2 * self.bounds[k]
elif b.pos[k] > self.bounds[k]:
b.pos[k] -= 2 * self.bounds[k]


def get_state_arrays(self):
return np.array([b.pos for b in self.boids]), np.array([b.vel for b in self.boids])