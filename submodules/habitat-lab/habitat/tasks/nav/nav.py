#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from typing import Any, Dict, List, Optional, Tuple

import attr

try:
    import magnum as mn
except ModuleNotFoundError:
    pass
import math
import os

import numpy as np
import squaternion
from gym import spaces
from gym.spaces.box import Box

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import ActionSpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war, maps

try:
    import habitat_sim
    from habitat_sim.bindings import RigidState
    from habitat_sim.physics import VelocityControl

    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat.sims.habitat_simulator.habitat_simulator import (
        HabitatSimDepthSensor,
        HabitatSimRGBSensor,
    )
    SIM_IMPORTED = True
except ImportError:
    SIM_IMPORTED = False

from .spot_utils.raibert_controller import (
    Raibert_controller_turn,
    Raibert_controller_turn_stable,
)

cv2 = try_cv2_import()

MAP_THICKNESS_SCALAR: int = 128


def merge_sim_episode_config(sim_config: Config, episode: Episode) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if episode.start_position is not None and episode.start_rotation is not None:
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(default=None, validator=not_none_validator)
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim

        self._goal_format = getattr(config, "GOAL_FORMAT", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR"]

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1] / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor
class ImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[self._rgb_sensor_uuid]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2**32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(episode)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "heading"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array([-agent_position[2], agent_position[0]], dtype=np.float32)
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(config, "MAX_DETECTION_RADIUS", 2.0)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )


@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        # print('SUCCESS: ', task.robot_wrapper.robot_dist_to_goal)
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if distance_to_target == 999.99:
            task.is_stop_called = True
            self._metric = 0.0
        else:
            if (
                hasattr(task, "is_stop_called")
                and task.is_stop_called  # type: ignore
                and distance_to_target < task.robot_wrapper.robot_dist_to_goal
            ):
                self._metric = 1.0
            else:
                self._metric = 0.0


@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(self._start_end_episode_distance, self._agent_episode_distance)
        )


@registry.register_measure
class SCT(SPL):
    r"""Success weighted by Completion Time"""

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "sct"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._num_steps_taken = 0
        self._was_last_success = False
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        if not ep_success or not self._was_last_success:
            self._num_steps_taken += 1
        self._was_last_success = ep_success

        oracle_time = self._start_end_episode_distance / self._config.HOLONOMIC_VELOCITY
        agent_time = self._num_steps_taken * self._config.TIME_STEP
        self._metric = ep_success * (oracle_time / max(oracle_time, agent_time))


@registry.register_measure
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(self._start_end_episode_distance, self._agent_episode_distance)
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._metric is None:
            self._metric = {"count": 0, "is_collision": False}
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count: Optional[int] = None
        self._map_resolution = config.MAP_RESOLUTION
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.DRAW_BORDER,
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(goal.position, maps.MAP_TARGET_POINT_INDICATOR)
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(sem_scene.objects[object_id].id.split("_")[-1]) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = sem_scene.objects[object_id].aabb.sizes / 2.0
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            _shortest_path_points = self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
            self._shortest_path_points = [
                maps.to_grid(p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim)
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=2.0):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        self._draw_goals_view_points(episode)
        self._draw_goals_aabb(episode)
        self._draw_goals_positions(episode)

        self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR)

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim),
            )


@registry.register_measure
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                    episode,
                )
                if distance_to_target == np.inf:
                    distance_to_target = 999.99
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target


@registry.register_measure
class EpisodeDistance(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "episode_distance"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        pass


@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.MOVE_FORWARD)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.TURN_LEFT)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.TURN_RIGHT)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "STOP"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_stop_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore


@registry.register_task_action
class LookUpAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.LOOK_UP)


@registry.register_task_action
class LookDownAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.LOOK_DOWN)


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: List[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()  # type: ignore

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task_action
class VelocityAction(SimulatorTaskAction):
    name: str = "VELOCITY_CONTROL"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        config = kwargs["config"]
        self.min_lin_vel, self.max_lin_vel = config.LIN_VEL_RANGE
        self.min_ang_vel, self.max_ang_vel = config.ANG_VEL_RANGE
        self.min_hor_vel, self.max_hor_vel = config.HOR_VEL_RANGE

        self.min_abs_lin_speed = config.MIN_ABS_LIN_SPEED
        self.min_abs_ang_speed = config.MIN_ABS_ANG_SPEED
        self.has_hor_vel = self.min_hor_vel != 0.0 and self.max_hor_vel != 0.0
        self.min_abs_hor_speed = config.MIN_ABS_HOR_SPEED

        self.must_call_stop = config.MUST_CALL_STOP
        self.time_step = config.TIME_STEP

        # For acceleration penalty
        self.prev_ang_vel = 0.0

        self.ctrl_freq = config.CTRL_FREQ
        self.dt = 1.0 / self.ctrl_freq

        # Noise
        self.noise_mean_x = config.NOISE_MEAN_X
        self.noise_mean_y = config.NOISE_MEAN_Y
        self.noise_var_x = config.NOISE_VAR_X
        self.noise_var_y = config.NOISE_VAR_Y
        self.noise_var_t = config.NOISE_VAR_T
        self.noise_mean_t = config.NOISE_MEAN_T
    @property
    def action_space(self):
        action_dict = {
            "linear_velocity": spaces.Box(
                low=np.array([self.min_lin_vel]),
                high=np.array([self.max_lin_vel]),
                dtype=np.float32,
            ),
            "angular_velocity": spaces.Box(
                low=np.array([self.min_ang_vel]),
                high=np.array([self.max_ang_vel]),
                dtype=np.float32,
            ),
        }

        if self.has_hor_vel:
            action_dict["horizontal_velocity"] = spaces.Box(
                low=np.array([self.min_hor_vel]),
                high=np.array([self.max_hor_vel]),
                dtype=np.float32,
            )

        return ActionSpace(action_dict)

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore
        self.prev_ang_vel = 0.0
        self.num_actions = 0
        self.num_collisions = 0

    def _reset_robot(self, task, agent_pos, agent_rot):
        task.robot_id.transformation = mn.Matrix4.from_(
            agent_rot.__matmul__(
                task.robot_wrapper.rotation_offset
            ).rotation(),  # 3x3 rotation
            agent_pos + task.robot_wrapper.robot_spawn_offset,  # translation vector
        )  # 4x4 homogenous transform

        # Spawn robot to agent location
        if task.robot_wrapper.name != "Locobot":
            # Set joints to default positions
            task.robot_wrapper.robot_specific_reset()

            # Reset joint motor controllers
            for i in range(12):
                task.robot_id.update_joint_motor(
                    i,
                    habitat_sim.physics.JointMotorSettings(
                        task.robot_wrapper._initial_joint_positions[
                            i
                        ],  # position_target
                        task.robot_wrapper.pos_gain,  # position_gain
                        0,  # velocity_target
                        task.robot_wrapper.vel_gain,  # velocity_gain
                        task.robot_wrapper.max_impulse,  # max_impulse
                    ),
                )

            # Reset raibert controller
            self.raibert_controller.set_init_state(task.robot_wrapper.calc_state())

    def get_curr_rigid_state(self):
        agent_state = self._sim.get_agent_state()
        normalized_quaternion = np.normalized(agent_state.rotation)
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )
        return current_rigid_state

    def put_text(self, task, agent_observations, lin_vel, hor_vel, ang_vel):
        try:
            robot_rigid_state = task.robot_id.rigid_state
            img = np.copy(agent_observations["rgb"])
            font = cv2.FONT_HERSHEY_PLAIN
            vel_text = "PA. Vx: {:.2f}, Vy: {:.2f}, Vt: {:.2f}".format(
                lin_vel, hor_vel, ang_vel
            )
            robot_state_text = "Robot state: {:.2f}, {:.2f}, {:.2f}".format(
                robot_rigid_state.translation.x,
                robot_rigid_state.translation.y,
                robot_rigid_state.translation.z,
            )
            dist_to_goal_text = "Dist2Goal: {:.2f}".format(
                task.measurements.measures["distance_to_goal"].get_metric()
            )
            cv2.putText(img, vel_text, (10, 20), font, 1, (0, 0, 0), 1)
            cv2.putText(img, robot_state_text, (10, 60), font, 1, (0, 0, 0), 1)
            cv2.putText(img, dist_to_goal_text, (10, 80), font, 1, (0, 0, 0), 1)
            agent_observations["rgb"] = img
        except:
            pass

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        lin_vel: float,
        ang_vel: float,
        hor_vel: Optional[float] = 0.0,
        time_step: Optional[float] = None,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            lin_vel: between [-1,1], scaled according to
                             config.LIN_VEL_RANGE
            ang_vel: between [-1,1], scaled according to
                             config.ANG_VEL_RANGE
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """
        curr_rs = task.robot_id.transformation
        if time_step is None:
            time_step = self.time_step

        # Convert from [-1, 1] to [0, 1] range
        lin_vel = (lin_vel + 1.0) / 2.0
        ang_vel = (ang_vel + 1.0) / 2.0
        hor_vel = (hor_vel + 1.0) / 2.0

        # Scale actions
        lin_vel = self.min_lin_vel + lin_vel * (self.max_lin_vel - self.min_lin_vel)
        ang_vel = self.min_ang_vel + ang_vel * (self.max_ang_vel - self.min_ang_vel)
        ang_vel = np.deg2rad(ang_vel)
        hor_vel = self.min_hor_vel + hor_vel * (self.max_hor_vel - self.min_hor_vel)

        called_stop = (
            abs(lin_vel) < self.min_abs_lin_speed
            and abs(ang_vel) < self.min_abs_ang_speed
            and abs(hor_vel) < self.min_abs_hor_speed
        )
        if (
            self.must_call_stop
            and called_stop
            or not self.must_call_stop
            and task.measurements.measures["distance_to_goal"].get_metric()
            < task.robot_wrapper.robot_dist_to_goal
        ):
            task.is_stop_called = True  # type: ignore
            return self._sim.get_observations_at(
                position=None,
                rotation=None,
            )

        # TODO: Sometimes the rotation given by get_agent_state is off by 1e-4
        # in terms of if the quaternion it represents is normalized, which
        # throws an error as habitat-sim/habitat_sim/utils/validators.py has a
        # tolerance of 1e-5. It is thus explicitly re-normalized here.

        # Convert from np.quaternion to mn.Quaternion
        if not self.has_hor_vel:
            hor_vel = 0.0
        self.vel_control.linear_velocity = np.array([-hor_vel, 0.0, -lin_vel])
        self.vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])
        agent_state = self._sim.get_agent_state()

        normalized_quaternion = np.normalized(agent_state.rotation)
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )

        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )
        start_rigid_state = current_rigid_state
        # manually integrate the rigid state
        goal_rigid_state = self.vel_control.integrate_transform(
            time_step, current_rigid_state
        )

        """Check if point is on navmesh"""
        goal_rigid_state.translation.x += np.random.normal(loc=self.noise_mean_x, scale=np.sqrt(self.noise_var_x))
        goal_rigid_state.translation.z += np.random.normal(loc=self.noise_mean_y, scale=np.sqrt(self.noise_var_y))
        ang = np.arctan2(goal_rigid_state.rotation.vector.y, goal_rigid_state.rotation.scalar) * 2 + \
              np.random.normal(loc=np.deg2rad(self.noise_mean_t), scale=np.sqrt(np.deg2rad(self.noise_var_t)))
        goal_rigid_state.rotation = mn.Quaternion.rotation(mn.Rad(ang), mn.Vector3(0, 1, 0))

        final_position = self._sim.pathfinder.try_step_no_sliding(  # type: ignore
            agent_state.position, goal_rigid_state.translation
        )
        # Check if a collision occured
        dist_moved_before_filter = (
            goal_rigid_state.translation - agent_state.position
        ).dot()
        dist_moved_after_filter = (final_position - agent_state.position).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the
        # filter is _less_ than the amount moved before the application of the
        # filter.
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        if collided:
            agent_observations = self._sim.get_observations_at()

            # agent_observations = self._sim.get_observations_at()
            self._sim._prev_sim_obs["collided"] = True  # type: ignore
            agent_observations["hit_navmesh"] = True
            agent_observations["moving_backwards"] = False
            agent_observations["moving_sideways"] = False
            agent_observations["ang_accel"] = 0.0
            if kwargs.get("num_steps", -1) != -1:
                agent_observations["num_steps"] = kwargs["num_steps"]

            self.prev_ang_vel = 0.0
            self.put_text(task, agent_observations, lin_vel, hor_vel, ang_vel)
            return agent_observations

        """See if goal state causes interpenetration with surroundings"""

        # Calculate next rigid state off agent rigid state

        goal_mn_quat = mn.Quaternion(
            goal_rigid_state.rotation.vector, goal_rigid_state.rotation.scalar
        )
        agent_T_global = mn.Matrix4.from_(
            goal_mn_quat.to_matrix(), goal_rigid_state.translation
        )
        robot_rotation_offset = mn.Matrix4.rotation(
            mn.Rad(0.0),
            mn.Vector3((1.0, 0.0, 0.0))
            # mn.Rad(-np.pi / 2.0), mn.Vector3((1.0, 0.0, 0.0))
        ).rotation()
        robot_translation_offset = mn.Vector3(task.robot_wrapper.robot_spawn_offset)
        robot_T_agent = mn.Matrix4.from_(
            robot_rotation_offset, robot_translation_offset
        )

        robot_T_global = robot_T_agent @ agent_T_global
        robot_rotation_offset = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2.0),
            mn.Vector3((1.0, 0.0, 0.0)),
        ) @ mn.Matrix4.rotation(
            mn.Rad(np.pi / 2.0),
            mn.Vector3((0.0, 0.0, 1.0)),
        )
        robot_T_global = robot_T_global @ robot_rotation_offset
        task.robot_id.transformation = robot_T_global

        collided = self._sim.contact_test(task.robot_id.object_id)
        if collided:
            task.robot_id.transformation = curr_rs
            agent_observations = self._sim.get_observations_at()
            self._sim._prev_sim_obs["collided"] = True  # type: ignore
            agent_observations["hit_navmesh"] = True
            agent_observations["moving_backwards"] = False
            agent_observations["moving_sideways"] = False
            agent_observations["ang_accel"] = 0.0
            if kwargs.get("num_steps", -1) != -1:
                agent_observations["num_steps"] = kwargs["num_steps"]

            self.prev_ang_vel = 0.0
            self.put_text(task, agent_observations, lin_vel, hor_vel, ang_vel)

            return agent_observations

        final_position = goal_rigid_state.translation

        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        agent_observations = self._sim.get_observations_at(
            position=final_position,
            rotation=final_rotation,
            keep_agent_at_new_pose=True,
        )
        # TODO: Make a better way to flag collisions
        self._sim._prev_sim_obs["collided"] = collided  # type: ignore
        agent_observations["hit_navmesh"] = collided
        agent_observations["moving_backwards"] = lin_vel < 0
        agent_observations["moving_sideways"] = abs(hor_vel) > self.min_abs_hor_speed
        agent_observations["ang_accel"] = (ang_vel - self.prev_ang_vel) / self.time_step
        if kwargs.get("num_steps", -1) != -1:
            agent_observations["num_steps"] = kwargs["num_steps"]

        self.prev_ang_vel = ang_vel
        self.put_text(task, agent_observations, lin_vel, hor_vel, ang_vel)

        return agent_observations


@registry.register_task_action
class DynamicVelocityAction(VelocityAction):
    name: str = "DYNAMIC_VELOCITY_CONTROL"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Joint motor gains
        config = kwargs["config"]
        self.time_per_step = config.TIME_PER_STEP

    @property
    def action_space(self):
        action_dict = {
            "linear_velocity": spaces.Box(
                low=np.array([self.min_lin_vel]),
                high=np.array([self.max_lin_vel]),
                dtype=np.float32,
            ),
            "angular_velocity": spaces.Box(
                low=np.array([self.min_ang_vel]),
                high=np.array([self.max_ang_vel]),
                dtype=np.float32,
            ),
        }

        if self.has_hor_vel:
            action_dict["horizontal_velocity"] = spaces.Box(
                low=np.array([self.min_hor_vel]),
                high=np.array([self.max_hor_vel]),
                dtype=np.float32,
            )

        return ActionSpace(action_dict)

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        super().reset(task=task, *args, **kwargs)
        agent_pos = kwargs["episode"].start_position
        agent_rot = kwargs["episode"].start_rotation

        self.start_height = agent_pos[1]
        squat = np.normalized(np.quaternion(agent_rot[3], *agent_rot[:3]))

        agent_rot = mn.Matrix4.from_(
            mn.Quaternion(squat.imag, squat.real).to_matrix(),
            mn.Vector3(0.0, 0.0, 0.0),
        )  # 4x4 homogenous transform with no translation

        # Raibert controller
        if task.robot_wrapper.name != "Locobot":
            self.raibert_controller = Raibert_controller_turn(
                control_frequency=self.ctrl_freq,
                num_timestep_per_HL_action=self.time_per_step,
                robot=task.robot_wrapper.name,
            )

        self._reset_robot(task, agent_pos, agent_rot)

        # Settle for 15 seconds to allow robot to land on ground and be still
        # self._sim.step_physics(15)

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        lin_vel: float,
        ang_vel: float,
        hor_vel: Optional[float] = 0.0,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            lin_vel: between [-1,1], scaled according to
                             config.LIN_VEL_RANGE
            ang_vel: between [-1,1], scaled according to
                             config.ANG_VEL_RANGE
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """

        # Convert from [-1, 1] to [0, 1] range
        lin_vel = (lin_vel + 1.0) / 2.0
        ang_vel = (ang_vel + 1.0) / 2.0
        hor_vel = (hor_vel + 1.0) / 2.0

        # Scale actions
        lin_vel = self.min_lin_vel + lin_vel * (self.max_lin_vel - self.min_lin_vel)
        ang_vel = self.min_ang_vel + ang_vel * (self.max_ang_vel - self.min_ang_vel)
        ang_vel = np.deg2rad(ang_vel)
        hor_vel = self.min_hor_vel + hor_vel * (self.max_hor_vel - self.min_hor_vel)
        if not self.has_hor_vel:
            hor_vel = 0.0

        called_stop = (
            abs(lin_vel) < self.min_abs_lin_speed
            and abs(ang_vel) < self.min_abs_ang_speed
            and abs(hor_vel) < self.min_abs_hor_speed
        )
        if (
            self.must_call_stop
            and called_stop
            or not self.must_call_stop
            and task.measurements.measures["distance_to_goal"].get_metric()
            < task.robot_wrapper.robot_dist_to_goal
        ):
            task.is_stop_called = True  # type: ignore
            return self._sim.get_observations_at(
                position=None,
                rotation=None,
            )

        # TODO: Sometimes the rotation given by get_agent_state is off by 1e-4
        # in terms of if the quaternion it represents is normalized, which
        # throws an error as habitat-sim/habitat_sim/utils/validators.py has a
        # tolerance of 1e-5. It is thus explicitly re-normalized here.

        ## STEP ROBOT
        if task.robot_wrapper.name != "Locobot":
            ## QUADRUPEDS:
            robot_start_rigid_state = task.robot_id.rigid_state
            for i in range(int(1 / self.time_step)):
                state = task.robot_wrapper.calc_state()
                if not self.has_hor_vel:
                    hor_vel = 0.0
                lin_hor = np.array([lin_vel, hor_vel])
                latent_action = self.raibert_controller.plan_latent_action(
                    state, lin_hor, target_ang_vel=ang_vel
                )
                self.raibert_controller.update_latent_action(state, latent_action)

                for i in range(self.time_per_step):
                    raibert_action = self.raibert_controller.get_action(state, i + 1)
                    task.robot_wrapper.apply_robot_action(raibert_action)
                    self._sim.step_physics(self.dt)
                    state = task.robot_wrapper.calc_state()
        else:
            ## LOCOBOT
            self.vel_control.linear_velocity = np.array([hor_vel, 0.0, -lin_vel])
            self.vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

            self._sim.step_physics(self.dt)

        final_robot_rigid_state = task.robot_id.rigid_state

        final_position = final_robot_rigid_state.translation

        final_rotation = mn.Quaternion.from_matrix(
            task.robot_id.transformation.__matmul__(
                task.robot_wrapper.rotation_offset.inverted()
            ).rotation()
        )
        final_rotation = [*final_rotation.vector, final_rotation.scalar]

        z_fall = final_position[1] < self.start_height + 0.1

        if z_fall and task.robot_wrapper.name != "Locobot":
            print("fell over")
            task.is_stop_called = True
            agent_observations = self._sim.get_observations_at(
                position=None,
                rotation=None,
            )

            agent_observations["fell_over"] = True
            return agent_observations

        agent_observations = self._sim.get_observations_at(
            position=final_position,
            rotation=final_rotation,
            keep_agent_at_new_pose=True,
        )

        # TODO: Make a better way to flag collisions
        agent_observations["moving_backwards"] = lin_vel < 0
        agent_observations["moving_sideways"] = abs(hor_vel) > self.min_abs_hor_speed
        agent_observations["ang_accel"] = (ang_vel - self.prev_ang_vel) / self.dt

        if kwargs.get("num_steps", -1) != -1:
            agent_observations["num_steps"] = kwargs["num_steps"]

        self.prev_ang_vel = ang_vel
        contacts = self._sim.get_physics_contact_points()
        num_contacts = self._sim.get_physics_num_active_contact_points()
        contacting_feet = set()
        for c in contacts:
            for link in [c.link_id_a, c.link_id_b]:
                contacting_feet.add(task.robot_id.get_link_name(link))

        if num_contacts > 5:
            self._sim._prev_sim_obs["collided"] = True  # type: ignore
            agent_observations["hit_navmesh"] = True
            self.num_collisions += 1
        self.put_text(task, agent_observations, lin_vel, hor_vel, ang_vel)

        return agent_observations


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_task(name="InteractiveNav-v0")
class InteractiveNavigationTask(NavigationTask):
    def reset(self, episode):
        self._sim.reset_objects(episode)
        observations = super().reset(episode)
        return observations


if SIM_IMPORTED:
    @registry.register_sensor
    class SpotLeftRgbSensor(HabitatSimRGBSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_left_rgb"


    @registry.register_sensor
    class SpotRightRgbSensor(HabitatSimRGBSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_right_rgb"


    @registry.register_sensor
    class SpotGraySensor(HabitatSimRGBSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_gray"

        ## Hack to get Spot cameras resized to 256,256 after concatenation
        def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
            return spaces.Box(
                low=0,
                high=255,
                shape=(256, 128, 1),
                dtype=np.float32,
            )

        def get_observation(self, sim_obs):
            obs = sim_obs.get(self.uuid, None)
            assert isinstance(obs, np.ndarray)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (128, 256))
            obs = obs.reshape([*obs.shape[:2], 1])
            return obs


    @registry.register_sensor
    class SpotLeftGraySensor(SpotGraySensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_left_gray"


    @registry.register_sensor
    class SpotRightGraySensor(SpotGraySensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_right_gray"


    @registry.register_sensor
    class SpotDepthSensor(HabitatSimDepthSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_depth"

        ## Hack to get Spot cameras resized to 256,256 after concatenation
        def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
            return spaces.Box(
                low=self.min_depth_value,
                high=self.max_depth_value,
                shape=(256, 128, 1),
                dtype=np.float32,
            )

        def get_observation(self, sim_obs):
            obs = sim_obs.get(self.uuid, None)
            assert isinstance(obs, np.ndarray)
            obs[obs > self.config.MAX_DEPTH] = 0.0  # Spot blacks out far pixels
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)
            obs = cv2.resize(obs, (128, 256))

            obs = np.expand_dims(obs, axis=2)  # make depth observation a 3D array
            if self.config.NORMALIZE_DEPTH:
                # normalize depth observation to [0, 1]
                obs = (obs - self.config.MIN_DEPTH) / (
                    self.config.MAX_DEPTH - self.config.MIN_DEPTH
                )

            return obs


    @registry.register_sensor
    class SpotLeftDepthSensor(SpotDepthSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_left_depth"


    @registry.register_sensor
    class SpotRightDepthSensor(SpotDepthSensor):
        def _get_uuid(self, *args, **kwargs):
            return "spot_right_depth"

