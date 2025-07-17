import math
import os
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame

FPS = 60
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 100
SPEED_SHOULDER = 7
SPEED_ELBOW = 7
SPEED_WRIST = 7

INITIAL_RANDOM = 5

ARM_L = 50 / SCALE
ARM_W = 4 / SCALE
HAND_L = 25 / SCALE
HAND_W = 4 / SCALE

HOOP_MIN_DIST = 8
HOOP_MAX_DIST = 9

VIEWPORT_W = 800
VIEWPORT_H = 600

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=circleShape(radius=0.5),
    density=5.0,
    friction=0.0,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

UPPER_ARM_FD = fixtureDef(
    shape=polygonShape(box=(ARM_L / 2, ARM_W / 2)),
    density=1.0,
    friction=0.5,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

FOREARM_FD = fixtureDef(
    shape=polygonShape(box=(ARM_W / 2, ARM_L / 2)),
    density=1.0,
    friction=0.5,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

HAND_FD = fixtureDef(
    shape=polygonShape(box=(HAND_L / 2, HAND_W / 2)),
    density=1.0,
    friction=5.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if self.env.ball in bodies:
            for body in bodies:
                if hasattr(body, 'userData') and body.userData == 'hoop_sensor':
                    self.env.ball_in_hoop = True

    def EndContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if self.env.ball in bodies:
            for body in bodies:
                if hasattr(body, 'userData') and body.userData == 'hoop_sensor':
                    self.env.ball_in_hoop = False
                    # Set the flag only when ball exits the hoop
                    if self.env.ball.linearVelocity.y < 0 and self.env.ball.position[1] > 8:
                        self.env.shot_made = True

class BasketballArm(gym.Env, EzPickle):

    def _add_hoop(self, position=(12,8)):
        x, y= position 
        #backboard
        self.backboard = self.world.CreateStaticBody(
            position=(x+0.5, y+1),
            shapes=polygonShape(box=(0.2, 1.5))
        )
        self.backboard.color1=(150,150,150)
        self.backboard.color2=(100,100,100)
        self.drawlist.append(self.backboard)
        #rim
        self.rim = self.world.CreateStaticBody(
            position = (x,y),
            fixtures=fixtureDef(
                shape=polygonShape(box=(1.1, 0.1)),
                isSensor=True
                )
        )

        self.rim.color1 = (255,0,0)
        self.rim.color2 = (200,0,0)
        self.rim.userData = 'hoop_sensor'
        
        self.drawlist.append(self.rim)

        #supports

        self.support = self.world.CreateStaticBody(
            position=(x-1.15 , y + 0.15),
            shapes=polygonShape(box=(0.05, 0.2))
        )
        self.support.color1 = (255, 0, 0)
        self.support.color2 = (200, 0, 0)
        self.drawlist.append(self.support)

    def _add_ball(self, position=(10, 10), radius=0.5):
        ball_fixture = fixtureDef(
            shape=circleShape(radius=radius),
            density=1.0,
            friction=0.3,
            restitution=0.6,  # bounciness
        )
        self.ball = self.world.CreateDynamicBody(
            position=position,
            fixtures=ball_fixture,
        )
        self.ball.color1 = (255, 165, 0) #inside color
        self.ball.color2 = (204, 102, 0) #border color
        self.drawlist.append(self.ball)
    """
    ## Description
    This is a simple 4-joint walker robot environment.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gymnasium/envs/box2d/bipedal_walker.py
    ```

    ## Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.

    ## Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ## Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ## Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ## Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ## Arguments

    To use the _hardcore_ environment, you need to specify the `hardcore=True`:

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<BipedalWalker<BipedalWalker-v3>>>>>

    ```

    ## Version History
    - v3: Returns the closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: str | None = None):
        EzPickle.__init__(self, render_mode)
        self.isopen = True

        self.world = Box2D.b2World()
        self.terrain: list[Box2D.b2Body] = []
        self.hull: Box2D.b2Body | None = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -20,
                -20,
                -20,
                -20,
                # -20,
                # -20,
                -20,
                -20,
                # -20,
                # -20,
                -math.pi,
                -5.0,
                -20,
                -20,
                -math.pi,
                -5.0,
                -20,
                -20,
                -math.pi,
                -5.0,
                -20,
                -20,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                20,
                20,
                20,
                20,
                # 20,
                # 20,
                20,
                20,
                # 20,
                # 20,
                math.pi,
                5.0,
                20,
                20,
                math.pi,
                5.0,
                20,
                20,
                math.pi,
                5.0,
                20,
                20,
            ]
        ).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        self.render_mode = render_mode
        self.screen: pygame.Surface | None = None
        self.clock = None

        self.max_x = 0
        self.max_y = 0
        self.prev_action = np.zeros(3)
        self.shot_made = False
        self.ball_in_hoop = False
        self.closest_distance = np.inf

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for arm in self.arms:
            self.world.DestroyBody(arm)
        self.arms = []
        self.joints = []
        self.world.DestroyBody(self.ball)
        self.ball = None
        self.world.DestroyBody(self.backboard)
        self.backboard = None
        self.world.DestroyBody(self.rim)
        self.rim = None
        self.world.DestroyBody(self.support)
        self.support = None
        

    def _generate_terrain(self):
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        

        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0

        self._generate_terrain()
        print(self.terrain)

        self.init_x = TERRAIN_STEP * TERRAIN_STARTPAD
        self.init_y = TERRAIN_HEIGHT + 2 * ARM_L
        
        self.hull = self.world.CreateStaticBody(
            position=(self.init_x, self.init_y), fixtures=HULL_FD,
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)

        self.arms: list[Box2D.b2Body] = []
        self.joints: list[Box2D.b2RevoluteJoint] = []
        upper_arm = self.world.CreateDynamicBody(
            position=(self.init_x + ARM_L / 2, self.init_y),
            angle=0,
            fixtures=UPPER_ARM_FD,
        )
        upper_arm.color1 = (255, 0, 0)
        upper_arm.color2 = (255, 0, 0)
        rjd = revoluteJointDef(
            bodyA=self.hull,
            bodyB=upper_arm,
            localAnchorA=(0, 0),
            localAnchorB=(-ARM_L / 2, 0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=0,
            lowerAngle=-np.pi / 2,
            upperAngle=np.pi / 3,
        )
        self.arms.append(upper_arm)
        self.joints.append(self.world.CreateJoint(rjd))

        forearm = self.world.CreateDynamicBody(
            position=(self.init_x + ARM_L, self.init_y + ARM_L / 2),
            angle=0,
            fixtures=FOREARM_FD,
        )
        forearm.color1 = (0, 255, 0)
        forearm.color2 = (0, 255, 0)
        rjd = revoluteJointDef(
            bodyA=upper_arm,
            bodyB=forearm,
            localAnchorA=(ARM_L / 2, 0),
            localAnchorB=(0, -ARM_L / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=0,
            lowerAngle=-np.pi / 2,
            upperAngle=np.pi / 3,
        )
        self.arms.append(forearm)
        self.joints.append(self.world.CreateJoint(rjd))

        hand = self.world.CreateDynamicBody(
            position=(self.init_x + ARM_L - HAND_L / 2, self.init_y + ARM_L),
            angle=0,
            fixtures=HAND_FD,
        )
        hand.color1 = (0, 0, 255)
        hand.color2 = (0, 0, 255)
        rjd = revoluteJointDef(
            bodyA=forearm,
            bodyB=hand,
            localAnchorA=(0, ARM_L / 2),
            localAnchorB=(HAND_L / 2, 0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=MOTORS_TORQUE,
            motorSpeed=0,
            lowerAngle=-np.pi,
            upperAngle=0,
        )
        self.arms.append(hand)
        self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.arms + [self.hull]

        initial_ball_position = (self.init_x + ARM_L - HAND_L / 2, self.init_y + ARM_L + 0.5)
        self._add_ball(position=initial_ball_position, radius=0.5)
        self.hoop_position = (self.init_x + np.random.uniform(HOOP_MIN_DIST, HOOP_MAX_DIST), self.init_y + 4)
        # self.hoop_position = (self.init_x + 9, self.init_y + 5)
        self._add_hoop(position=self.hoop_position)
        self.closest_distance = math.sqrt((initial_ball_position[0] - self.hoop_position[0]) ** 2 + (initial_ball_position[1] - self.hoop_position[1]) ** 2)
        self.target_y = self.init_y + 7
        self.closest_y_distance = self.target_y - initial_ball_position[1]
        self.target_x = self.hoop_position[0]
        self.closest_x_distance = self.target_x - initial_ball_position[0]

        self.max_x = 0
        self.max_y = 0
        self.prev_action = np.zeros(3)
        self.shot_made = False
        self.ball_in_hoop = False

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_SHOULDER * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_ELBOW * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_WRIST * np.clip(action[2], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_SHOULDER * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_ELBOW * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_WRIST * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        # vel = self.hull.linearVelocity

        state = [
            self.hoop_position[0],
            self.hoop_position[1],
            self.ball.position[0],
            self.ball.position[1],
            # self.ball.position[0] - self.hoop_position[0],
            # self.ball.position[1] - self.hoop_position[1],
            self.ball.linearVelocity[0],
            self.ball.linearVelocity[1],
            # self.arms[2].position[0] - self.ball.position[0],
            # self.arms[2].position[1] - self.ball.position[1],
            self.joints[0].angle,
            self.joints[0].speed / SPEED_SHOULDER,
            self.arms[0].position[0],
            self.arms[0].position[1],
            self.joints[1].angle,
            self.joints[1].speed / SPEED_ELBOW,
            self.arms[1].position[0],
            self.arms[1].position[1],
            self.joints[2].angle,
            self.joints[2].speed / SPEED_WRIST,
            self.arms[2].position[0],
            self.arms[2].position[1],
        ]
        # assert len(state) == 24
        # print(len(state))
        assert len(state) == 18

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        ball_x, ball_y = self.ball.position[0], self.ball.position[1]
        reward = 0

        distance = math.sqrt((ball_x - self.hoop_position[0]) ** 2 + (ball_y - self.hoop_position[1]) ** 2)
        if distance < self.closest_distance:
            reward += 1.0 * (self.closest_distance - distance)
            self.closest_distance = distance
        
        y_distance = self.target_y - ball_y
        if y_distance < self.closest_y_distance and y_distance >= 0:
            reward += 0.8 * (self.closest_y_distance - y_distance)
            self.closest_y_distance = y_distance
        
        x_distance = self.target_x - ball_x
        if x_distance < self.closest_x_distance and x_distance >= 0:
            reward += 0.8 * (self.closest_x_distance - x_distance)
            self.closest_x_distance = x_distance

        # if ball_x > self.max_x:
        #     reward += ball_x - self.max_x
        #     self.max_x = ball_x
        # if ball_y > self.max_y:
        #     reward += ball_y - self.max_y
        #     self.max_y = ball_y

        action_change_penalty = np.sum((action - self.prev_action) ** 2)
        reward -= 0.1 * action_change_penalty  # scale to balance
        # for a in action:
            # reward -= 0.003 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
        
        if self.shot_made:
            reward += 50
            self.shot_made = False

        terminated = False
        if self.ball.position[1] <= TERRAIN_HEIGHT + 0.6:
            # reward += 100 * (self.max_x * 0.5 + self.max_y)
            terminated = True

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def make_env(record_video=False, video_folder="videos", video_length=1000):
    def _init():
        env = BasketballArm(render_mode='rgb_array')
        env.reset()
        env = Monitor(env)  # for logging
        return env

    env = DummyVecEnv([_init])

    if record_video:
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda x: x % 50000 == 0,
            video_length=video_length,
            name_prefix="basketball_arm_episode",
        )

    return env


def train_model(total_timesteps=501_000, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    env = make_env(record_video=True)
    eval_env = make_env(record_video=False)
    eval_callback = EvalCallback(eval_env,
                             log_path=log_dir, eval_freq=50000,
                             deterministic=True)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("ppo_basketball_arm")

    return model


def evaluate_and_render(model_path="ppo_basketball_arm", episodes=5):
    env = BasketballArm(render_mode="human")

    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1} reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    model = train_model()

    # Evaluate and visualize
    evaluate_and_render()
