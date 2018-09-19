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

namespace flame {

    namespace stereo {

        class TestStereoEpipolar {
        public:
            TestStereoEpipolar() {
                width_ = 752;
                height_ = 480;

                image0_ = cv::imread("../data/1403636643463555584_left.png", CV_LOAD_IMAGE_GRAYSCALE);
                image1_ = cv::imread("../data/1403636643463555584_right.png", CV_LOAD_IMAGE_GRAYSCALE);

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


        TEST(StereoEpipolarGeometryTest, project) {
            TestStereoEpipolar testStereoEpipolar;

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

            std::vector<cv::KeyPoint> u_inf_s;
            std::vector<cv::KeyPoint> u_zero_s;
            std::vector<cv::KeyPoint> u_cmp_s;
            cv::RNG rng;
            for (auto kp : kp0) {
                cv::Point2f u_inf, u_zero, u_cmp;
                epipolarGeometry.maxDepthProjection(kp.pt, &u_inf);
                epipolarGeometry.minDepthProjection(kp.pt, &u_zero);

                float depth = rng.uniform(1.0, 2.0);
                u_cmp = epipolarGeometry.project(kp.pt, 1.0/ depth );
                cv::KeyPoint kp_inf, kp_zero, kp_cmp;
                kp_inf.pt = u_inf;
                kp_zero.pt = u_zero;
                kp_cmp.pt = u_cmp;
                u_inf_s.push_back(kp_inf);
                u_zero_s.push_back(kp_zero);
                u_cmp_s.push_back(kp_cmp);
//                std::cout<< "kp_inf: " << kp_inf.pt.x << " " << kp_inf.pt.y
//                         << " "<< kp_zero.pt.x << " " <<kp_zero.pt.y << std::endl;

                cv::line(kp_canvas1, u_zero, u_inf,cv::Scalar( 0,225, 225));
            }


            cv::drawKeypoints(kp_canvas1, u_inf_s, kp_canvas1, cv::Scalar( 0,225, 0));
            cv::drawKeypoints(kp_canvas1, u_zero_s, kp_canvas1, cv::Scalar( 0,225, 225));
            cv::drawKeypoints(kp_canvas1, u_cmp_s, kp_canvas1, cv::Scalar( 225,0, 225));


//
            cv::imshow("left", kp_canvas);
            cv::imshow("right", kp_canvas1);

            while( true){
                int c = cv::waitKey(10);
                if (c == 27) break;
            }

        }
    }

}