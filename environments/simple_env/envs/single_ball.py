import os
import random
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data

from stable_baselines3 import PPO
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

class SingleBallEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observation_dim=8, action_dim=4, max_episode_len=2000):
        # Connect the pybullet physics engine(server) with GUI mode
        p.connect(p.GUI)

        # Set the initial position and orientation of the view in GUI
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,     # distance from eye to camera target position
            cameraYaw=0,            # camera yaw angle (in degrees) left/right
            cameraPitch=-40,        # camera pitch angle (in degrees) up/down
            cameraTargetPosition=[0.55, -0.35, 0.2]  # the camera focus point
        )
        
        # Define the observation space in ranging -1 to 1
        # observation: Cartesian position of the gripper (3) + two fingers joint positions (2)
        # + Cartesian position of the object (3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(observation_dim, ), dtype=np.float32)

        # Define the action space in ranging -1 to 1
        # action: Cartesian position of the gripper (3) + a joint variable for both fingers (1). 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim, ), dtype=np.float32)

        # Initialize step length
        self.step_counter = 0
        self.max_episode_len = max_episode_len

        self.object = None
        self.basket = None

    def reset(self, seed=None, options=None):
        # Path configuration
        urdf_path = pybullet_data.getDataPath()
        plane_path = os.path.join(urdf_path, "plane.urdf")
        table_path = os.path.join(urdf_path, "table/table.urdf")
        object_path = os.path.join(urdf_path, "random_urdfs/000/000.urdf")
        basket_path = os.path.join(urdf_path, "tray/traybox.urdf")
        robot_path = os.path.join(urdf_path, "franka_panda/panda.urdf")
        
        # Initial states of robot and objects
        self.step_counter = 0
        robot_joint_position = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]

        # Reset simulation and environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Stop rendering
        p.setGravity(0, 0, -10)

        # Load objects into the simulation
        p.loadURDF(plane_path, basePosition=[0, 0, -0.65])
        p.loadURDF(table_path, basePosition=[0.5, 0, -0.65])
        self.panda_id = p.loadURDF(robot_path, useFixedBase=True)
        
        # Reset robot joint states
        for i in range(7):
            p.resetJointState(self.panda_id, i, robot_joint_position[i])
        p.resetJointState(self.panda_id, 9, 0.08)
        p.resetJointState(self.panda_id, 10, 0.08)

        # Load object (ball) and basket
        object_position = [0.6, 0, 0.05]
        self.object = p.loadURDF(object_path, basePosition=object_position)
        p.changeVisualShape(self.object, -1, rgbaColor=[1, 0, 0, 1])

        basket_position = [0.3, 0, 0.05]
        self.basket = p.loadURDF(basket_path, basePosition=basket_position, globalScaling=0.5)
        p.changeVisualShape(self.basket, -1, rgbaColor=[0, 1, 0, 1])

        # Close the gripper to hold the object
        p.resetJointState(self.panda_id, 9, 0.02)
        p.resetJointState(self.panda_id, 10, 0.02)

        # Get observation info
        observation = self._get_observation()

        # Start rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation, {}

    def step(self, action):
        # Set single-step rendering for immediate feedback
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        # Get the current position of the gripper
        current_pose = p.getLinkState(self.panda_id, 11)
        current_position = current_pose[0]

        # Set the goal position of the gripper
        dv = 0.1
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        finger_position = action[3]
        goal_position = [current_position[0] + dx,
                        current_position[1] + dy,
                        current_position[2] + dz]
        goal_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

        # Calculate inverse kinematics for the goal position
        ik_joint_states = p.calculateInverseKinematics(
            bodyUniqueId=self.panda_id,
            endEffectorLinkIndex=11,
            targetPosition=goal_position,
            targetOrientation=goal_orientation)
        goal_arm_joint_states = ik_joint_states[:7]
        goal_states = list(goal_arm_joint_states) + [finger_position, finger_position]

        # Apply the calculated joint positions and finger position to the robot
        joints_to_be_controlled = list(range(7)) + [9, 10]
        p.setJointMotorControlArray(
            bodyUniqueId=self.panda_id,
            jointIndices=joints_to_be_controlled,
            controlMode=p.POSITION_CONTROL,
            targetPositions=goal_states
        )

        # Step the simulation
        p.stepSimulation()

        # Get the state of the object and the robot
        observation = self._get_observation()

        # Determine 'reward' and whether the task is 'done'
        reward = self._compute_reward()
        done = self._is_done()

        self.step_counter += 1
        if self.step_counter > self.max_episode_len:
            done = True

        # Set 'info'
        info = {}

        # New termination and truncation logic
        terminated = done
        truncated = self.step_counter > self.max_episode_len

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.7, 0.0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(960) / 720,
            nearVal=0.1,
            farVal=100.0
        )
        width, height, rgb_pixels, depth_pixels, _ = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(depth_pixels, dtype=np.float32)
        depth_array = np.reshape(depth_array, (height, width))
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            return rgb_array
        elif mode == 'depth_array':
            return depth_array

    def close(self):
        p.disconnect()

    def _get_observation(self):
        # Gripper state
        gripper_state = p.getLinkState(self.panda_id, 11)
        gripper_pos = gripper_state[0]

        gripper_joint_state = p.getJointStates(self.panda_id, [9, 10])
        gripper_joint_pos = [gripper_joint_state[0][0], gripper_joint_state[1][0]]

        # Object state
        object_pos, object_orient = p.getBasePositionAndOrientation(self.object)
        object_pos = np.array(object_pos)

        observation = np.concatenate([gripper_pos, gripper_joint_pos, object_pos])
        return observation

    def _compute_reward(self):
        reward = 0
        object_pos = p.getBasePositionAndOrientation(self.object)[0]
        basket_pos = p.getBasePositionAndOrientation(self.basket)[0]
        distance = np.linalg.norm(np.array(object_pos[:3]) - np.array(basket_pos[:3]))
        if distance < 0.1:
            reward += 1
        return reward

    def _is_done(self):
        object_pos = p.getBasePositionAndOrientation(self.object)[0]
        basket_pos = p.getBasePositionAndOrientation(self.basket)[0]
        distance = np.linalg.norm(np.array(object_pos[:3]) - np.array(basket_pos[:3]))
        if distance < 0.1:
            return True
        return False

    def test(self):
        self.reset()
        self.render()
        for _ in range(1000000):
            action = self.action_space.sample()
            observation, reward, done, info = self.step(action)
            self.render()
            if done:
                print("Task completed")
                break

def plot_results(log_folder, title='Learning Curve'):
    """
    Plot the results using matplotlib
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = np.convolve(y, np.ones(50)/50, mode='valid')  # smooth out the curve
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.savefig(os.path.join(log_folder, 'learning_curve.png'))
    plt.close()

if __name__ == "__main__":

    # env = SingleBallEnv()
    # env.test()
    # env.close()

    model_path = '/home/yeonarom/robot/ros2/panda_gym_ws/panda-rl/agents'
    callback_log_path = os.path.join(model_path, "models", "single_ball-v1", "temp")
    log_path = os.path.join(model_path, "logs", "single_ball-v1")

    env = SingleBallEnv()
    env = Monitor(env, os.path.join(callback_log_path, 'monitor.csv'))

    total_iterations = 1_000_000
    n_steps = 2048
    batch_size = 512
    n_epochs = 128
    policy_kwargs = {'net_arch': [512, 512]}

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps = 2048,
        batch_size = 512,
        n_epochs = 128,
        verbose=1,
        tensorboard_log=log_path,)
    '''
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        gamma=0.95,
        # ----------------------------
        learning_rate=0.001,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        # ----------------------------
        tensorboard_log=log_path,
        verbose=1,
        seed=256,
        device='auto',                
    )
    '''
    # Train the model
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=callback_log_path
    )
    model.learn(
        total_timesteps=total_iterations,
        callback=callback,
        log_interval=4,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # Save the trained model
    model.save(os.path.join(model_path, "models", "single_ball-v1"))
    print("Model saved to {}".format('single_ball-v1'))

    # Plot the learning curve
    plot_results(callback_log_path, title='Learning Curve')

    del model
    env.close()
