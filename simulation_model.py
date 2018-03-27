import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
from numpy.random import normal
import random
import copy
import math
import pybullet_data
import time
import cv2
from pprint import pprint
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

def add_one_obj_to_scene(num,pos=None, orientation = None, global_scaling=1):
    """
    Inputs:
        num: int, serial number of the object
        pos: 3D vector, pose of the object
        orientation: quaternion, orientation of the object
    Output:
        obj: pybullet object
    """
    assert num<1000, "random object number cannot exceed 1000!"
    assert pos is not None, "pos cannot be empty!"
    if orientation is None:
        obj = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num,num),pos,globalScaling=global_scaling)
    else:
        obj = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num,num),pos,orientation,globalScaling=global_scaling)
    return obj

def add_objs_to_scene(nums,poses,orientations=None):
    """
    Inputs:
        nums: list of ints, serial numbers of the objects
        poses: list of 3D vector, a list of poses of the objects
        orientations: list of quaternions, optional, a list of quaternions of the objects
    Outputs:
        objs: a list of pybullet objects
    """
    assert len(nums)==len(poses), "number of objects should match number of poses!"
    if orientations is not None:
        assert len(orientations)==len(nums), "size of orientations should match size of objects!"
        return [add_one_obj_to_scene(num,pos,orientation) for num,pos,orientation in zip(nums,poses,orientations)]
    else:
        return [add_one_obj_to_scene(num,pos) for num,pos in zip(nums,poses)]
        
def add_random_objs_to_scene(size,pos_mean=[0,0],pos_height=1,orientation_mean=[0,0,0,1]):
    nums = [random.randint(0,999) for _ in range(size)]
    poses = [[normal(0.6,0.12),normal(scale=0.16),normal(0.05,0.05)] for _ in range(size)]#container center position (0.6,0,0)
    orientations = [normal(size=(4)) for _ in range(size)]
    pprint(nums)
    print()
    pprint(poses)
    objs = add_objs_to_scene(nums,poses,orientations)
    # objs = add_objs_to_scene(nums,poses)
    return objs


def main():
    environment = KukaGymEnv(renders=True,isDiscrete=False, maxSteps = 10000000)
    randomObjs = add_random_objs_to_scene(10)	  
    motorsIds=[]
    motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
    motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
    motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    # dv = 0.01 
    # motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    done = False
    #According to the hand-eye coordination paper, the camera is mounted over the shoulder of the arm
    #TODO: find appropriate view matrix and projection matrix parameters
    cametaEyePosition = []
    cametaTargetPosition = []
    cametaUpVector = []
    left,right,bottom,top,near,far = 0,0,0,0,0,0

    img = p.getCameraImage(640,512)#640*512*3 
    print(type(img))
    print(len(img))
    print(type(img[0]))
    print(type(img[1]))
    #tuples
    print(type(img[2]))#1310720=640*512*4
    print(len(img[2]))
    print(type(img[3]))#327680=
    print(len(img[3]))
    print(type(img[4]))#327680=640*512
    print(len(img[4]))

    # cv2.imwrite("./test.jpeg",img)
    while (not done):   
        action=[]
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))        
        state, reward, done, info = environment.step2(action)
        obs = environment.getExtendedObservation()


    # physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    # p.setGravity(0,0,-10)
    # planeId = p.loadURDF("plane.urdf")
    
    # cubeStartPos = [0,0,1]
    # cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    # # randomObj = add_one_obj_to_scene(80, [2,2,0],cubeStartOrientation)
    # randomObjs = add_random_objs_to_scene(10)

    # boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
    # for i in range (10000):
    #     p.stepSimulation()
    #     time.sleep(1./240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
    # p.disconnect()
if __name__ == '__main__':
    main()
    