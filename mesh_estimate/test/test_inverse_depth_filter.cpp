#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "gtest/gtest.h"

#include "flame/stereo/epipolar_geometry.h"
#include "flame/stereo/inverse_depth_filter.h"
#include "flame/params.h"

namespace flame {

    namespace stereo {
        class TestInverseDepthFilter {
        public:
            TestInverseDepthFilter() {
                width_ = 752;
                height_ = 480;

                image0_ = cv::imread("../data/1403636659863555584_left.png", CV_LOAD_IMAGE_GRAYSCALE);
                image1_ = cv::imread("../data/1403636659863555584_right.png", CV_LOAD_IMAGE_GRAYSCALE);

                K0_<< 458.654, 0 , 367.215,
                        0, 457.296,  248.375,
                        0,0,1;

                K0cv_ = (cv::Mat_<float>(3,3) << 458.654, 0 , 367.215,
                        0, 457.296,  248.375,
                        0,0,1);

                K1_ << 457.587, 0,  379.999,
                        0,  456.134,255.238,
                        0,0,1;

                K1cv_ = (cv::Mat_<float>(3,3) << 457.587, 0,  379.999,
                        0,  456.134,255.238,
                        0,0,1);

                D0_ << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05;
                D0cv_  = (cv::Mat_<float>(4,1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

                D1_  << -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05;
                D1cv_  = (cv::Mat_<float>(4,1) << -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05);


                T_WC0_.matrix() << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                        0.0, 0.0, 0.0, 1.0;

                T_WC1_.matrix() << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                        0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                        0.0, 0.0, 0.0, 1.0;

                T_C1_C0_ = T_WC1_.inverse() * T_WC0_;


                distortImage();
            }

            void distortImage() {
                cv::undistort(image0_, distortedImage0_, K0cv_, D0cv_);
                cv::undistort(image1_, distortedImage1_, K1cv_, D1cv_);
            }

            Eigen::Matrix3f K0_;
            Eigen::Matrix3f K1_;
            Eigen::Vector4d D0_;
            Eigen::Vector4d D1_;

            cv::Mat K0cv_;
            cv::Mat K1cv_;
            cv::Mat D0cv_;
            cv::Mat D1cv_;

            Eigen::Isometry3d T_WC0_;
            Eigen::Isometry3d T_WC1_;
            Eigen::Isometry3d T_C1_C0_;

            int width_;
            int height_;

            cv::Mat image0_;
            cv::Mat distortedImage0_;
            cv::Mat image1_;
            cv::Mat distortedImage1_;
        };

        TEST(InverseDepthFilterTest, research) {
            TestInverseDepthFilter testStereoEpipolar;



            EpipolarGeometry<float> epipolarGeometry(testStereoEpipolar.K0_, testStereoEpipolar.K0_.inverse(),
                                                     testStereoEpipolar.K1_, testStereoEpipolar.K1_.inverse());
            Eigen::Matrix3d C = testStereoEpipolar.T_C1_C0_.linear();
            Eigen::Quaterniond quat_C1_C0(C);
            epipolarGeometry.loadGeometry(quat_C1_C0.cast<float>(),
                                          testStereoEpipolar.T_C1_C0_.translation().cast<float>());
            const int fast_th = 80;
            std::vector<cv::KeyPoint> kp0;
            cv::FAST(testStereoEpipolar.distortedImage0_, kp0, fast_th,true );
            cv::Mat kp_canvas, kp_canvas1;
            cv::cvtColor(testStereoEpipolar.distortedImage0_, kp_canvas, CV_GRAY2BGR);
            cv::cvtColor(testStereoEpipolar.distortedImage1_, kp_canvas1, CV_GRAY2BGR);
            cv::drawKeypoints(kp_canvas, kp0, kp_canvas, cv::Scalar(225, 0,0));


        }

    }
}