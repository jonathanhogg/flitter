"""
Flitter physics engine
"""

from loguru import logger

from ..model import Vector
from ..language import functions


class Particle:
    def __init__(self, number, node, dimensions, prefix, state):
        zero = Vector([0] * dimensions)
        self.position_key = Vector.compose([prefix, Vector(number)])
        self.velocity_key = Vector.compose([prefix, Vector('velocity'), Vector(number)])
        initial = Vector.coerce(node.get('initial', dimensions, float, zero))
        self.position = state[self.position_key] or initial
        self.velocity = state[self.velocity_key] or zero
        self.force = zero
        self.mass = node.get('mass', 1, float, 1)
        self.radius = node.get('radius', 1, float, 1)

    def update(self, delta, state):
        self.velocity += self.force / self.mass * delta
        self.position += self.velocity * delta
        state[self.position_key] = self.position
        state[self.velocity_key] = self.velocity


class Anchor(Particle):
    def __init__(self, number, node, dimensions, prefix, state):
        self.zero = Vector([0] * dimensions)
        self.position_key = Vector.compose([prefix, Vector(number)])
        self.velocity_key = Vector.compose([prefix, Vector('velocity'), Vector(number)])
        self.position = Vector.coerce(node.get('position', 3, float, self.zero))
        self.velocity = self.zero
        self.force = self.zero
        self.mass = 0
        self.radius = 0

    def update(self, delta, state):
        state[self.position_key] = self.position
        state[self.velocity_key] = self.zero


class DistanceConstraint:
    def __init__(self, node):
        if (fixed := node.get('fixed', 1, float)):
            self.minimum = fixed
            self.maximum = fixed
        else:
            self.minimum = node.get('min', 1, float)
            self.maximum = node.get('max', 1, float)
        self.from_particle = node.get('from', 1, int)
        self.to_particle = node.get('to', 1, int)
        self.weight = node.get('weight', 1, float, 1)

    def update_forces(self, particles):
        from_particle = particles.get(self.from_particle)
        to_particle = particles.get(self.to_particle)
        if from_particle and to_particle:
            vector = to_particle.position - from_particle.position
            distance = functions.hypot(vector)
            direction = vector / distance
            if self.minimum and distance < self.minimum:
                force = (self.weight * (self.minimum - distance)) * direction
                from_particle.force -= force
                to_particle.force += force
            elif self.maximum and distance > self.maximum:
                force = (self.weight * (distance - self.maximum)) * direction
                from_particle.force += force
                to_particle.force -= force


class DragConstraint:
    def __init__(self, node):
        self.coefficient = node.get('coefficient', 1, float, 1)

    def update_forces(self, particles):
        if self.coefficient:
            for particle in particles.values():
                force = self.coefficient * (1 + functions.absv(particle.velocity)) * particle.velocity
                particle.force -= force


class CollisionConstraint:
    def __init__(self, node):
        self.weight = node.get('weight', 1, float, 1)

    def update_forces(self, particles):
        if self.weight:
            particles = list(particles.values())
            n = len(particles)
            for i in range(n):
                for j in range(i):
                    from_particle = particles[i]
                    to_particle = particles[j]
                    if from_particle.radius and to_particle.radius:
                        min_distance = from_particle.radius + to_particle.radius
                        vector = to_particle.position - from_particle.position
                        distance = functions.hypot(vector)
                        direction = vector / distance
                        if distance < min_distance:
                            force = (self.weight * (min_distance - distance)) * direction
                            from_particle.force -= force
                            to_particle.force += force


class PhysicsSystem:
    def __init__(self):
        logger.debug("Create physics system")
        self.last_beat = None

    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, node, now):
        dimensions = node.get('dimensions', 1, int)
        if dimensions < 1:
            return
        state_prefix = node.get('state')
        if not state_prefix:
            return
        particles = {}
        constraints = []
        for child in node.children:
            if child.kind == 'anchor':
                number = child.get('number', 1, int)
                if number is not None:
                    particles[number] = Anchor(number, child, dimensions, state_prefix, engine.state)
            elif child.kind == 'particle':
                number = child.get('number', 1, int)
                if number is not None:
                    particles[number] = Particle(number, child, dimensions, state_prefix, engine.state)
            elif child.kind == 'distance':
                constraints.append(DistanceConstraint(child))
            elif child.kind == 'drag':
                constraints.append(DragConstraint(child))
            elif child.kind == 'collision':
                constraints.append(CollisionConstraint(child))
        for constraint in constraints:
            constraint.update_forces(particles)
        beat = engine.counter.beat_at_time(now)
        delta = min((beat - self.last_beat) if self.last_beat is not None else 0, 1/20)
        self.last_beat = beat
        for particle in particles.values():
            particle.update(delta, engine.state)


INTERACTOR_CLASS = PhysicsSystem
