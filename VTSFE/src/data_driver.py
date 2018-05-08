# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import c3d
import pdb

from klepto.archives import file_archive


class Data_driver():

    DATA_PARAMS = {
        "path": "./data/7x10_actions/XSens/xml",
        "nb_frames": 70,
        "nb_samples_per_mov": 10,
        "mov_types": [
            # AnDy mocaps
            "bent_fw",
            "bent_fw_strongly",
            "kicking",
            "lifting_box",
            "standing",
            "walking",
            "window_open",

            # CMU mocaps
            # "taichi",
            # "kicking",
            # "walking",
            # "punching"
        ],
        "normalization_as_3d": False,
        "relative_movement": True,
        "use_center_of_mass": True
    }

    chen_db_path = './data/5x1_actions/ChenDB/mocap_dataset.mat'

    hard_segments = [
        24, 25, 26,
        27, 28, 29,
        30, 31, 32,
        36, 37, 38,
        39, 40, 41,
        42, 43, 44,
        48, 49, 50,
        51, 52, 53,
        54, 55, 56,
        60, 61, 62,
        63, 64, 65,
        66, 67, 68
    ]

    hard_joints = [
        21, 22, 23,
        24, 25, 26,
        33, 34, 35,
        36, 37, 38,
        42, 43, 44,
        45, 46, 47,
        54, 55, 56,
        57, 58, 59
    ]

    joints = [
        "Pelvis/L5",
        "L5/L3",
        "L3/T12",
        "T12/T8",
        "T8/Neck",
        "Neck/Head",
        "T8/RightShoulder",
        "RightShoulder/RightUpperArm",
        "RightUpperArm/RightForeArm",
        "RightForeArm/RightHand",
        "T8/LeftShoulder",
        "LeftShoulder/LeftUpperArm",
        "LeftUpperArm/LeftForeArm",
        "LeftForeArm/LeftHand",
        "Pelvis/RightUpperLeg",
        "RightUpperLeg/RightLowerLeg",
        "RightLowerLeg/RightFoot",
        "RightFoot/RightToe",
        "Pelvis/LeftUpperLeg",
        "LeftUpperLeg/LeftLowerLeg",
        "LeftLowerLeg/LeftFoot",
        "LeftFoot/LeftToe",
    ]

    segments = [
        "Pelvis",
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "RightToe",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToe",
    ]

    def __init__(self, params={}):
        self.__dict__.update(self.DATA_PARAMS, **params)
        self.scaling_factors = []
        joints = []
        for joint in self.joints:
            for dim in ["_x", "_y", "_z"]:
                joints.append(joint+dim)
        self.joints = joints
        segments = []
        for segment in self.segments:
            for dim in ["_x", "_y", "_z"]:
                segments.append(segment+dim)
        self.segments = segments


    def save_data(self, path, index, data):
        db = file_archive(path)
        db[index] = data
        db.dump()


    def read_data(self, path, index):
        db = file_archive(path)
        db.load()
        return db[index]


    def unit_bounds_rescaling(self, sample, keep_zero_mean=True):
        """ Bounds values to [-1, 1]
        """
        self.scaling_factors = []
        rescaled_sample = np.copy(sample)
        sample_shape = rescaled_sample.shape

        if keep_zero_mean:
            max_val = np.amax(np.absolute(rescaled_sample))
            rescaled_sample /= max_val
            self.scaling_factors.append(max_val)
        else:
            # normalization to range [0,1]
            max_val = np.amax(rescaled_sample)
            min_val = np.amin(rescaled_sample)
            rescaled_sample -= min_val
            rescaled_sample /= (max_val-min_val)
            # scaling to [-1,1]
            rescaled_sample *= 2.
            rescaled_sample -= 1.
            self.scaling_factors.append((max_val, min_val))

        print("Data rescaled to unit bounds [-1, 1].")
        return rescaled_sample


    def undo_unit_bounds_rescaling(self, sample, keep_zero_mean=True):
        if self.scaling_factors:
            rescaled_sample = np.copy(sample)
            sample_shape = rescaled_sample.shape

            if keep_zero_mean:
                rescaled_sample *= self.scaling_factors[0]
            else:
                rescaled_sample += 1.
                rescaled_sample /= 2.
                rescaled_sample *= (self.scaling_factors[0][0] - self.scaling_factors[0][1])
                rescaled_sample += self.scaling_factors[0][1]

            return rescaled_sample
        else:
            return sample


    def parse(self):
        if self.data_source == "ChenDB":
            self.parse_CheKarSma2016_article()
        if self.data_source == "MVNX":
            self.parse_MVNX()


    def parse_CheKarSma2016_article(self):
        self.path = self.chen_db_path
        mat_file = sio.loadmat(self.path)
        db = mat_file["mocap_dataset"][0]
        data = []
        labels = []

        self.data_types = ["jointAngle"]
        self.hard_dimensions = self.hard_joints
        self.dim_names = self.joints
        self.mov_indices = {}
        self.mov_types = []
        for k in range(len(db)):
            self.mov_types.append(str(k))

        self.nb_samples_per_mov = 1

        for k, sample in enumerate(db):
            # sample shape = [nb_frames, n_input]
            nb_frames = len(sample)
            sample = sample[:self.nb_frames]

            data.append(np.array(sample, dtype=np.float32))
            labels.append(str(k))

        self.data = np.array(data, dtype=np.float32)
        # data normalization as in article, i.e. zero mean for every joint and in range [-1, 1]
        joint_means = np.mean(self.data, axis=0)
        joint_means = np.mean(joint_means, axis=0)
        for i, m in enumerate(joint_means):
            self.data[:, :, i] -= m

        # bounds values to [-1, 1]
        if self.unit_bounds:
            self.data = self.unit_bounds_rescaling(self.data)

        shape_val = self.data.shape[1:]
        sh = (-1, 1,) + shape_val
        # data shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
        self.data = np.reshape(self.data, sh)

        self.data_labels = np.array(labels)
        for i, mov in enumerate(self.mov_types):
            self.mov_indices[mov] = labels.index(mov)

        self.input_dim = self.data.shape[-1]


    def parse_CMU(self, folderpath=None, unit_bounds=True, relative_movement=True):
        """
        [DEPRECATED]
        Parses data from c3d files from the CMU Database, as in the article VAE-DMP.
        These files contains frames of movements, each containing 41 marker points in 5 dimensions (3 firsts for 3D coordinates)
        """

        data = []
        labels = []
        filepaths = []
        samples_per_mov_count = np.zeros((len(self.mov_types)), dtype=np.int)
        self.mov_indices = {}

        for f in sorted(listdir(folderpath)):
            path = join(folderpath, f)
            if isfile(path):
                mov_type = f[:f.index("-")]
                if mov_type in self.mov_types:
                    mov_type_index = self.mov_types.index(mov_type)
                    if samples_per_mov_count[mov_type_index] < self.nb_samples_per_mov:
                        samples_per_mov_count[mov_type_index] += 1
                        filepaths.append((path, mov_type))

        for filepath, label in filepaths:
            print(filepath+" : reading c3d file...")
            reader = c3d.Reader(open(filepath, 'rb'))

            frames = []
            for index, points, analog in reader.read_frames():
                # shape = [nb_frames, nb_markers, nb_coordinates]
                frames.append(points)

            nb_frames = len(frames)
            frames_3d = np.array(frames)[:, :, :3]

            # data normalization as in article, i.e. zero mean and range in [-1, 1])

            # If we want values to be relative to the barycenter of all positions at a given frame
            if relative_movement:
                m = np.mean(frames_3d, axis=1)
                for i in range(nb_frames):
                    for d in range(3):
                        frames_3d[i, :, d] -= m[i, d]
            else:
                m = np.mean(frames_3d, axis=0)
                m = np.mean(m, axis=0)
                for d in range(3):
                    frames_3d[:, :, d] -= m[d]

            # subsampling step computing
            frame_mod = nb_frames % self.nb_frames
            if frame_mod == 0:
                sampling_step = int(nb_frames / self.nb_frames)
            else:
                sampling_step = int((nb_frames-frame_mod) / self.nb_frames)

            sample_data = []
            step = 0
            # subsampling
            for i in range(self.nb_frames):
                values = []

                for j, marker in enumerate(frames_3d[step]):
                    values = np.concatenate([values, marker])
                sample_data.append(values)
                step += sampling_step

            labels.append(label)
            data.append(sample_data)
            print(filepath+" : cmu file parsed successfully.")

        # output data shape = [nb_mov*nb_samples_per_mov, nb_frames, 3*marker_count]
        self.data = np.array(data, dtype=np.float32)
        # bounds values to [-1, 1]
        if unit_bounds:
            self.data = self.unit_bounds_rescaling(self.data)

        self.data_labels = np.array(labels)
        self.mov_types = np.unique(labels)

        for mov in self.mov_types:
            self.mov_indices[mov] = labels.index(mov)


    def parse_MVNX(self):
        """ Parses all MVNX files in folderpath, searching for values in data_types for each frame of movement.
        Each file in folderpath is a sample.
        You also may define the number of samples per movement type, as well as the number of movement types.

        data_types is a list of strings specifying the wanted data types.
        data types available :
            - position
            - orientation
            - velocity
            - acceleration
            - angularVelocity
            - angularAcceleration
            - sensorAngularVelocity
            - sensorOrientation
            - jointAngle
            - jointAngleXZY
            - centerOfMass
        """

        ns = {"mvnx": "http://www.xsens.com/mvn/mvnx"}

        if len(self.data_types) == 1:
            if "position" in self.data_types:
                self.hard_dimensions = self.hard_segments
                self.dim_names = self.segments
            else:
                self.hard_dimensions = self.hard_joints
                self.dim_names = self.joints


        def extract(filepaths):
            data = []
            labels = []

            # initialize data : first dim for mov_type, second dim for mov sample, then dims for actual data
            for i in range(len(self.mov_types)):
                data.append([])

            for filepath, label in filepaths:
                
                tree = ET.parse(filepath)
                root = tree.getroot()

                # frames of one sample
                # frames shape = [nb_frames, n_input]
                frames = root.findall("./mvnx:subject/mvnx:frames/mvnx:frame[@type='normal']", ns)
                nb_frames = len(frames) #real number of data

                if nb_frames < self.nb_frames:
                    print(filepath+" : mvnx file has "+nb_frames+" frames, which is less than "+self.nb_frames+" frames. File ignored.")
                    continue

                frame_mod = nb_frames % self.nb_frames
                if frame_mod == 0:
                    sampling_step = int(nb_frames / self.nb_frames)
                else:
                    sampling_step = int((nb_frames-frame_mod) / self.nb_frames)

                com = []
                sample_data = []
                step = 0
                # subsampling
                for i in range(self.nb_frames):
                    values = []
                    for data_type in self.data_types:
                        values += frames[step].find("mvnx:"+data_type, ns).text.split()
                    if self.use_center_of_mass:
                        # center of mass per frame and per 3D dimension.
                        com.append(frames[step].find("mvnx:centerOfMass", ns).text.split())
                    # sample_data shape = [self.nb_frames, n_input]
                    sample_data.append(values)
                    step += sampling_step

                sample_data = np.array(sample_data, dtype=np.float32)

                if self.normalization_as_3d:
                    # reshape to make 3D coordinates appear
                    sample_data = sample_data.reshape([self.nb_frames, -1, 3])

                    # data normalization as in article, i.e. zero mean and range in [-1, 1])

                    # If we want values to be relative to the barycenter of all positions at a given frame
                    if self.relative_movement:
                        if self.use_center_of_mass:
                            m = np.array(com, dtype=np.float32)
                            for i in range(self.nb_frames):
                                for d in range(3):
                                    sample_data[i, :, d] -= m[i, d]
                        else:
                            # mean of all values. One mean value per frame and per 3D dimension.
                            m = np.mean(sample_data, axis=1)
                            for i in range(self.nb_frames):
                                for d in range(3):
                                    sample_data[i, :, d] -= m[i, d]
                    else:
                        # mean of all values. One mean value per source and per 3D dimension.
                        m = np.mean(sample_data, axis=0)
                        # mean of all values. One mean value per 3D dimension.
                        m = np.mean(m, axis=0)
                        for d in range(3):
                            sample_data[:, :, d] -= m[d]

                labels.append(label)
                # data shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
                data[self.mov_types.index(label)].append(sample_data)
                print(filepath+" : mvnx file parsed successfully.")

            data = np.array(data, dtype=np.float32)
            if not self.normalization_as_3d:
                # data normalization as in article, i.e. zero mean for every joint and in range [-1, 1]
                joint_means = np.mean(data, axis=0)
                joint_means = np.mean(joint_means, axis=0)
                joint_means = np.mean(joint_means, axis=0)
                for i, m in enumerate(joint_means):
                    data[:, :, :, i] -= m

            # bounds values to [-1, 1]
            if self.unit_bounds:
                data = self.unit_bounds_rescaling(data)

            # output data shape = [nb_mov_types, nb_samples_per_mov, nb_frames, segment_count]
            data = data.reshape([len(self.mov_types), self.nb_samples_per_mov, self.nb_frames, -1])
            return data, labels

        filepaths = []
        
        samples_per_mov_count = [0]*len(self.mov_types)
        self.mov_indices = {}

        for f in sorted(listdir(self.path)):
            path = join(self.path, f)
            if isfile(path):
                mov_type = f[:f.index("-")]
                if mov_type in self.mov_types:
                    mov_type_index = self.mov_types.index(mov_type)
                    if samples_per_mov_count[mov_type_index] < self.nb_samples_per_mov:
                        samples_per_mov_count[mov_type_index] += 1
                        filepaths.append((path, mov_type))

        data, labels = extract(filepaths)
        for i, mov in enumerate(self.mov_types):
            self.mov_indices[mov] = labels.index(mov)
        self.data = np.array(data, dtype=np.float32)
        self.data_labels = np.array(labels)
        self.input_dim = self.data.shape[-1]
        


    def saveData(self,nbLS=69):
        
        for actionType in range(7):
            #print("saving data "+str(actionType))
            try:
              #  print("try to create: "+"./data/observations/"+self.data_labels[(actionType*10)+1])
                os.mkdir("./data/observations/"+self.data_labels[(actionType*10)+1])
            except OSError:
               # print("Error during creation: "+"./data/observations/"+self.data_labels[(actionType*10)+1])
                pass
            for innx in range(10):
                f = open("./data/observations/"+self.data_labels[(actionType*10)+1]+"/record"+str(innx)+".txt", "w+")
                for vb in range(0,70):
                    nameString = ''
                    
                    for nbstring in range(nbLS-1):
                        nameString += str(self.data[actionType,innx,vb,nbstring])+"\t"
                    nameString +=str(self.data[actionType,innx,vb,nbLS-1])+"\n"
                    f.write(nameString)
                #print("saving test "+str(innx))
                f.close()
    
    
    def split(self, nb_blocks, nb_samples, train_proportion=0, test_proportion=0, eval_proportion=0):
        if train_proportion < 0 or test_proportion < 0 or eval_proportion < 0:
            return None
        norm_sum = train_proportion + test_proportion + eval_proportion
        if norm_sum == 0:
            return None
        block_size = int(nb_samples / nb_blocks)
        nb_train_blocks = int(train_proportion * nb_blocks / norm_sum)
        nb_test_blocks = int(test_proportion * nb_blocks / norm_sum)
        nb_eval_blocks = nb_blocks - nb_train_blocks - nb_test_blocks
        return block_size, nb_train_blocks, nb_test_blocks, nb_eval_blocks


    def to_set(self, data):
        # set shape = [nb_samples, nb_frames, n_input]
        return np.concatenate(data, axis=0)


    def get_whole_data_set(self, shuffle_dataset=True):
        return self.get_data_set(range(self.nb_samples_per_mov), shuffle_dataset=shuffle_dataset, shuffle_samples=False)[0]


    def get_data_set(self, sample_indices, shuffle_samples=True, shuffle_dataset=True):
        data_copy = np.copy(self.data)
        if sample_indices is not None:
            remains_indices = []
            for index in range(self.nb_samples_per_mov):
                if index not in sample_indices:
                    remains_indices.append(index)
        if shuffle_samples:
            for mov_type in data_copy:
                np.random.shuffle(mov_type)
        samples = np.reshape(data_copy[:, sample_indices], [-1, self.nb_frames, self.input_dim])
        remains = np.reshape(data_copy[:, remains_indices], [-1, self.nb_frames, self.input_dim])
        if shuffle_dataset:
            np.random.shuffle(samples)
            np.random.shuffle(remains)
        return samples, remains
