#include "flame/mesh_estimator.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <okvis/Estimator.hpp>

namespace flame {
    MeshEstimator::MeshEstimator(okvis::Estimator* estimator, int width, int height,
                                 const Matrix3f& K0, const Matrix3f& K0inv,
                                 const Vector4f& distort0,
                                 const Matrix3f& K1, const Matrix3f& K1inv,
                                 const Vector4f& distort1,
                                 const Params& parameters):
            params_(parameters),
            poseframe_subsample_factor_(6),
            estimator_(estimator){

        cv::eigen2cv(K0, K0cv_);
        cv::eigen2cv(distort0, D0cv_);

        cv::eigen2cv(K1, K1cv_);
        cv::eigen2cv(distort1, D1cv_);

        sensor_ = std::make_shared<flame::Flame>(estimator,
                width,
                                                 height,
                                                 K0,
                                                 K0inv,
                                                 K1,
                                                 K1inv,
                                                 params_);

    }

    void MeshEstimator::processFrame( const double time, int32_t img_id,
                      const okvis::kinematics::Transformation& T_WC0,
                                      const cv::Mat& img_gray0,
                                      const okvis::kinematics::Transformation& T_WC1,
                                      const cv::Mat& img_gray1,bool isKeyframe) {
//
        /*==================== Process image ====================*/
        cv::Mat img_gray_undist0;
        cv::undistort(img_gray0, img_gray_undist0, K0cv_, D0cv_);

        cv::Mat img_gray_undist1;
        cv::undistort(img_gray1, img_gray_undist1, K1cv_, D1cv_);

        bool is_poseframe = isKeyframe;
        bool update_success = false;
        update_success = sensor_->update(time, img_id,
                T_WC0, img_gray_undist0,
                                        T_WC1, img_gray_undist1,
                                             is_poseframe);


        sensor_->updateKeyframePose();

        Image3b wireImage = sensor_->getDebugImageWireframe();
//        Image3b wireImage = sensor_->getDebugImageFeatures();
        cv::imshow("wireImage", wireImage);

        Image3b depthImage = sensor_->getDebugImageInverseDepthMap();
        cv::imshow("depthImage", depthImage);
        cv::waitKey(1);


    }


    void MeshEstimator::estimateMesh(okvis::MultiFramePtr& multiFramePtr) {


    }
}