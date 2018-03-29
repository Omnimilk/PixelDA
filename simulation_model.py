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
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv

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
    poses = [[normal(0.6,0.1),normal(scale=0.16),normal(0.05,0.05)] for _ in range(size)]#container center position (0.6,0,0)
    orientations = [normal(size=(4)) for _ in range(size)]
    objs = add_objs_to_scene(nums,poses,orientations)
    # objs = add_objs_to_scene(nums,poses)
    return objs

def write_from_imgarr(imgarr,serial_number,path="sim_images/{0:0>6}.jpeg"):
    bgra =imgarr[2]#imgarr[3] depth image; imgarr[4] segmentation mask
    img = np.reshape(bgra, (512, 640, 4)).astype(np.uint8)#BGRA
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)#RGB
    cv2.imwrite(path.format(serial_number),img)

def write_from_npimg(npimg,serial_number,path="sim_images/{0:0>6}.jpeg"):
    img = np.reshape(npimg, (512, 640, 4)).astype(np.uint8)#BGRA
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)#RGB
    cv2.imwrite(path.format(serial_number),img)

def main_usingEnvOnly():
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
    # camEyePos = [0.03,0.236,0.54]
    # distance = 1.06
    # pitch=-56
    # yaw = 258
    # roll=0
    # upAxisIndex = 2
    # camInfo = p.getDebugVisualizerCamera()
    # viewMat = camInfo[2]
    # viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos,distance,yaw, pitch,roll,upAxisIndex)
    # projMatrix = camInfo[3]
    # print(viewMat)
    # print(projMatrix)
    viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
    projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
    write_from_imgarr(img_arr, 1)
    
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

def main():
    environment = KukaCamGymEnv(renders=True,isDiscrete=False)  
    randomObjs = add_random_objs_to_scene(10)
    motorsIds = []
    #motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
	#motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
	#motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
	#motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
	#motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
    dv = 1 
    motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))	
    done = False
    img_serial_num = 0
    step = 0
    snapshot_interval = 20
    while (not done):    
        action=[]
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        if step%snapshot_interval==0:
            #get cameta image
            viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
            projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
            img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
            write_from_imgarr(img_arr, img_serial_num)
            # np_img = environment.getExtendedObservation()#shape is not correct
            # write_from_npimg(np_img,img_serial_num)
            img_serial_num +=1
        step +=1
        state, reward, done, info = environment.step(action)
        obs = environment.getExtendedObservation()
if __name__ == '__main__':
    main()
    