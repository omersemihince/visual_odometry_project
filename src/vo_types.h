/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project, 
    covers the process of defining the basic objects that will be shared across the project files.

Structure is as fololws,
    Define pixel location of a landmark in the image as a 2D vector.
    Define the 3D location of a landmark in the world as a 3D vector -> triangulated point.
    Define the rotation and translation of the camera in SE(3).
    Define the ID of a landmark as an integer.
    Define complete observation of an image.
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

//Defining one detected landmark in one image with its pixel location and ID.
struct Observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //Allocate memory for Eigen types

    int landmark_id = -1; //ID of the landmark, -1 if not assigned yet; so they are invalid
    Eigen::Vector2f uv = Eigen::Vector2f::Zero(); //Pixel location of the landmark in the image (u,v)
};

//Defining one full observation of an image, which includes the image number and all the detected landmarks in that image together with camera pose.
struct Frame{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //Allocate memory for Eigen types

    int seq = -1; //Number of the image / sequence    
    
    std::vector<Observation> observations; //All the detected (visible) landmarks in that image
    
    Eigen::Isometry3f T_wc = Eigen::Isometry3f::Identity(); //Pose of the camera (transformation matrix from the world to the camera frame in SE(3) T_wc = [R_wc, t_wc; 0, 1] so p_c = T_wc * p_w)
    bool pose_valid = false; 
};

//Defining the Lanmark points (positions)
struct Landmark{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //Allocate memory for Eigen types

    int id = -1; //ID of the landmark, -1 if not assigned yet; so they are invalid
    Eigen::Vector3f p_world = Eigen::Vector3f::Zero(); //3D position of the landmark in the world frame

    std::vector<std::pair<int, Eigen::Vector2f>> observations; //List of pairs of (image number, pixel location) where this landmark is observed "history". This will be used for triangulation of the newly detected landmarks.

    bool initialized = false;
};

//Defining the Output of the epipolar_initializer.cpp
struct EpipolarInitResult{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //Allocate memory for Eigen types

    bool success = false; //Whether the initialization was successful or not.
    std::string message;

    Eigen::Isometry3f T_wc0 = Eigen::Isometry3f::Identity(); //Pose of the camera in instant Zero
    Eigen::Isometry3f T_wc1 = Eigen::Isometry3f::Identity(); //Pose of the camera in instant One
    
    std::unordered_map<int, Landmark> landmarks; //Initial map, unordered map is better for faster lookup in the future comparing to a list.
    
    int num_matches = 0; //# of landmarks with correspondences.
    int num_valid_triangulated = 0; //# of landmarks with valid triangulation.
};
