#include <okvis/estimate_pipeline.hpp>
#include <map>

#include <glog/logging.h>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>
#include <flame/mesh_estimator.hpp>
#include <flame/utils/frame.h>
#include <okvis/VioVisualizer.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>

namespace okvis {
    static const okvis::Duration temporal_imu_data_overlap(0.02);  // overlap of imu data before and after two consecutive frames [seconds]

    // Constructor.
    EstimatePipeline::EstimatePipeline(okvis::VioParameters& parameters, flame::Params mesh_parameter)
            : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
              imu_params_(parameters.imu),
              repropagationNeeded_(false),
              lastAddedImageTimestamp_(okvis::Time(0, 0)),
              estimator_(),
              frontend_(parameters.nCameraSystem.numCameras()),
              parameters_(parameters),
              meshParams(mesh_parameter){
        init();
    }

    EstimatePipeline::~EstimatePipeline() {

    }


    void EstimatePipeline::init() {
        assert(parameters_.nCameraSystem.numCameras() > 0);
        numCameras_ = parameters_.nCameraSystem.numCameras();
        numCameraPairs_ = 1;

        frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
        frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
        frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

        lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
        lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)


        estimator_.addImu(parameters_.imu);
        for (size_t i = 0; i < numCameras_; ++i) {
            // parameters_.camera_extrinsics is never set (default 0's)...
            // do they ever change?
            estimator_.addCamera(parameters_.camera_extrinsics);
            }

        // set up windows so things don't crash on Mac OS
        if(parameters_.visualization.displayImages){
//            for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
//                std::stringstream windowname;
//                windowname << "OKVIS camera " << im;
//                cv::namedWindow(windowname.str());
//            }
        }

        displayImages_.resize(numCameras_, cv::Mat());

        /*
         * Mesh estimator
         */

        double width = parameters_.nCameraSystem.cameraGeometry(0)->imageWidth();
        double height = parameters_.nCameraSystem.cameraGeometry(0)->imageHeight();
        Eigen::VectorXd intrinsic;
        parameters_.nCameraSystem.cameraGeometry(0)->getIntrinsics(intrinsic);
        double fx = intrinsic[0];
        double fy = intrinsic[1];
        double cx = intrinsic[2];
        double cy = intrinsic[3];
        double k1 = intrinsic[4];
        double k2 = intrinsic[5];
        double k3 = intrinsic[6];
        double k4 = intrinsic[7];

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = fx; K(1,1) = fy; K(0,2) = cx; K(1,2) = cy;
//
        Eigen::Vector4d distort;
        distort << k1,k2,k3,k4;


        parameters_.nCameraSystem.cameraGeometry(1)->getIntrinsics(intrinsic);
         fx = intrinsic[0];
         fy = intrinsic[1];
         cx = intrinsic[2];
         cy = intrinsic[3];
         k1 = intrinsic[4];
         k2 = intrinsic[5];
         k3 = intrinsic[6];
         k4 = intrinsic[7];

        Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
        K1(0,0) = fx; K1(1,1) = fy; K1(0,2) = cx; K1(1,2) = cy;
//
        Eigen::Vector4d distort1;
        distort1 << k1,k2,k3,k4;

        meshEstimatorPtr_
                = std::make_shared<flame::MeshEstimator>(&estimator_, width, height,
                                                         K.cast<float>(), K.inverse().cast<float>(),
                                                         distort.cast<float>(),
                                                         K1.cast<float>(), K1.inverse().cast<float>(),
                                                                 distort1.cast<float>(),
                                                                 meshParams);
    }


// Add a new image.
    bool EstimatePipeline::addImage(const okvis::Time & stamp,
                                 const cv::Mat & image0,
                                 const cv::Mat & image1,
                                 const std::vector<cv::KeyPoint> * keypoints,
                                 bool* /*asKeyframe*/) {

        if (lastAddedImageTimestamp_ > stamp
            && fabs((lastAddedImageTimestamp_ - stamp).toSec())
               > parameters_.sensors_information.frameTimestampTolerance) {
            LOG(ERROR)
                    << "Received image from the past. Dropping the image.";
            return false;
        }
        lastAddedImageTimestamp_ = stamp;

        /**************************   make frame ******************************/
        std::shared_ptr<okvis::MultiFrame> multiFrame;
        multiFrame = std::shared_ptr<okvis::MultiFrame>(
                new okvis::MultiFrame(parameters_.nCameraSystem,stamp,
                                      okvis::IdProvider::instance().newId()));
        multiFrame->setImage(0, image0);
        multiFrame->setImage(1, image1);

        /************************** detect and describe ******************************/

        okvis::kinematics::Transformation T_WS;
        okvis::Time lastTimestamp;
        okvis::SpeedAndBias speedAndBiases;
        // copy last state variables
        {

            T_WS = lastOptimized_T_WS_;
            speedAndBiases = lastOptimizedSpeedAndBiases_;
            lastTimestamp = lastOptimizedStateTimestamp_;
        }

        // -- get relevant imu messages for new state
        okvis::Time imuDataEndTime = multiFrame->timestamp()
                                     + temporal_imu_data_overlap;
        okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

        OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");



        okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                               imuDataEndTime);

        // if imu_data is empty, either end_time > begin_time or
        // no measurements in timeframe, should not happen, as we waited for measurements
        if (imuData.size() == 0) {
            return false;
        }

        if (imuData.front().timeStamp > multiFrame->timestamp()) {
            LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
            return false;
        }

        // get T_WC(camIndx) for detectAndDescribe()
        if (estimator_.numFrames() == 0) {
            // first frame ever
            bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
            {
                lastOptimized_T_WS_ = T_WS;
                lastOptimizedSpeedAndBiases_.setZero();
                lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
                lastOptimizedStateTimestamp_ = multiFrame->timestamp();
            }
            OKVIS_ASSERT_TRUE_DBG(Exception, success,
                                  "pose could not be initialized from imu measurements.");
            if (!success) {
                return false;
            }
        } else {
            // get old T_WS
            okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                                speedAndBiases, lastTimestamp,
                                                multiFrame->timestamp());
        }

        int border = meshParams.fparams.win_size;
        for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); i++) {
            okvis::kinematics::Transformation T_WC = T_WS
                                                     * (*parameters_.nCameraSystem.T_SC(i));

            // for brisk feature
            frontend_.detectAndDescribe(i, multiFrame, T_WC, nullptr);
        }

        /**************************  addState and match  ******************************/

        bool asKeyframe = false;
        if (estimator_.addStates(multiFrame, imuData, asKeyframe)) {
            lastAddedStateTimestamp_ = multiFrame->timestamp();
        } else {
            LOG(ERROR) << "Failed to add state! will drop multiframe.";
            return false;
        }

        // -- matching keypoints, initialising landmarks etc.
        estimator_.get_T_WS(multiFrame->id(), T_WS);
        frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,
                                                   map_, multiFrame, &asKeyframe);
        if (asKeyframe)
            estimator_.setKeyframe(multiFrame->id(), asKeyframe);

        // std::cout<< "okvis add landmark:  " << estimator_.numLandmarks() << std::endl;

        okvis::kinematics::Transformation T_WC0 = T_WS * (*parameters_.nCameraSystem.T_SC(0));
        okvis::kinematics::Transformation T_WC1 = T_WS * (*parameters_.nCameraSystem.T_SC(1));
        // todo(pang): give dense frame pose
        meshEstimatorPtr_->processFrame(multiFrame->timestamp().toSec(), static_cast<int32_t >(multiFrame->id()),
                                        T_WC0, multiFrame->image(0),
                                        T_WC1, multiFrame->image(1), asKeyframe);



        /***************  optimization and marginalisation ***************/
        okvis::Time deleteImuMeasurementsUntil(0, 0);

        OptimizationResults result;
        estimator_.optimize(parameters_.optimization.max_iterations, 2, false);

        // get timestamp of last frame in IMU window.
        // Need to do this before marginalization as it will be removed there (if not keyframe)
        if (estimator_.numFrames()
            > size_t(parameters_.optimization.numImuFrames)) {
            deleteImuMeasurementsUntil = estimator_.multiFrame(
                            estimator_.frameIdByAge(parameters_.optimization.numImuFrames))
                                                 ->timestamp() - temporal_imu_data_overlap;
        }

        estimator_.applyMarginalizationStrategy(
                parameters_.optimization.numKeyframes,
                parameters_.optimization.numImuFrames, result.transferredLandmarks);


        // now actually remove measurements
        deleteImuMeasurements(deleteImuMeasurementsUntil);

        // saving optimized state and saving it in OptimizationResults struct
        {

            estimator_.get_T_WS(multiFrame->id(), lastOptimized_T_WS_);
            estimator_.getSpeedAndBias(multiFrame->id(), 0,
                                       lastOptimizedSpeedAndBiases_);
            lastOptimizedStateTimestamp_ = multiFrame->timestamp();

            // if we publish the state after each IMU propagation we do not need to publish it here.

            result.T_WS = lastOptimized_T_WS_;
            result.speedAndBiases = lastOptimizedSpeedAndBiases_;
            result.stamp = lastOptimizedStateTimestamp_;
            result.onlyPublishLandmarks = false;

            estimator_.getLandmarks(result.landmarksVector);

            repropagationNeeded_ = true;
        }


        // adding further elements to result that do not access estimator.
        for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
            result.vector_of_T_SCi.push_back(
                    okvis::kinematics::Transformation(
                            *parameters_.nCameraSystem.T_SC(i)));
        }



        /***************  Visualization ***************/
        VioVisualizer::VisualizationData::Ptr visualizationDataPtr;

        if (parameters_.visualization.displayImages) {
            // fill in information that requires access to estimator.
            visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
                    new VioVisualizer::VisualizationData());
            visualizationDataPtr->observations.resize(multiFrame->numKeypoints());
            okvis::MapPoint landmark;
            okvis::ObservationVector::iterator it = visualizationDataPtr
                    ->observations.begin();
            for (size_t camIndex = 0; camIndex < multiFrame->numFrames();
                 ++camIndex) {
                for (size_t k = 0; k < multiFrame->numKeypoints(camIndex); ++k) {
                    OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
                    it->keypointIdx = k;
                    multiFrame->getKeypoint(camIndex, k, it->keypointMeasurement);
                    multiFrame->getKeypointSize(camIndex, k, it->keypointSize);
                    it->cameraIdx = camIndex;
                    it->frameId = multiFrame->id();
                    it->landmarkId = multiFrame->landmarkId(camIndex, k);
                    if (estimator_.isLandmarkAdded(it->landmarkId)) {
                        estimator_.getLandmark(it->landmarkId, landmark);
                        it->landmark_W = landmark.point;
                        if (estimator_.isLandmarkInitialized(it->landmarkId))
                            it->isInitialized = true;
                        else
                            it->isInitialized = false;
                    } else {
                        it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
                    }
                    ++it;
                }
            }
            visualizationDataPtr->keyFrames = estimator_.multiFrame(
                    estimator_.currentKeyframeId());
            estimator_.get_T_WS(estimator_.currentKeyframeId(),
                                visualizationDataPtr->T_WS_keyFrame);

            visualizationDataPtr->currentFrames = multiFrame;


            okvis::VioVisualizer visualizer_(parameters_);

            //visualizer_.showDebugImages(new_data);
            std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
            for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
                out_images[i] = visualizer_.drawMatches(visualizationDataPtr, i);
            }
            displayImages_ = out_images;
            display();
        }

        /***************  State callback ***************/

        if (fullStateCallback_ && !result.onlyPublishLandmarks) {
            fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                               result.omega_S);
        }



//            cv::imshow("image0", visualizationDataPtr->keyFrames->image(0));
//            cv::imshow("image1", visualizationDataPtr->keyFrames->image(1));
//            cv::waitKey(2);
        return true;
    }


// Add an IMU measurement.
    bool EstimatePipeline::addImuMeasurement(const okvis::Time & stamp,
                                          const Eigen::Vector3d & alpha,
                                          const Eigen::Vector3d & omega) {

        okvis::ImuMeasurement imu_measurement;
        imu_measurement.measurement.accelerometers = alpha;
        imu_measurement.measurement.gyroscopes = omega;
        imu_measurement.timeStamp = stamp;


        imuMeasurements_.push_back(imu_measurement);
        return true;

    }
    // trigger display (needed because OSX won't allow threaded display)
    void EstimatePipeline::display() {
        // draw
        for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
            std::stringstream windowname;
            windowname << "OKVIS camera " << im;
            cv::imshow(windowname.str(), displayImages_[im]);
        }
        cv::waitKey(1);
    }


// Get a subset of the recorded IMU measurements.
    okvis::ImuMeasurementDeque EstimatePipeline::getImuMeasurments(
            okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
        // sanity checks:
        // if end time is smaller than begin time, return empty queue.
        // if begin time is larger than newest imu time, return empty queue.
        if (imuDataEndTime < imuDataBeginTime
            || imuDataBeginTime > imuMeasurements_.back().timeStamp)
            return okvis::ImuMeasurementDeque();

        // get iterator to imu data before previous frame
        okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_
                .begin();
        okvis::ImuMeasurementDeque::iterator last_imu_package =
                imuMeasurements_.end();
        // TODO go backwards through queue. Is probably faster.
        for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
             ++iter) {
            // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
            if (iter->timeStamp <= imuDataBeginTime)
                first_imu_package = iter;

            // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
            if (iter->timeStamp >= imuDataEndTime) {
                last_imu_package = iter;
                // since we want to include this last imu measurement in returned Deque we
                // increase last_imu_package iterator once.
                ++last_imu_package;
                break;
            }
        }

        // create copy of imu buffer
        return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
    }

// Remove IMU measurements from the internal buffer.
    int EstimatePipeline::deleteImuMeasurements(const okvis::Time& eraseUntil) {
        if (imuMeasurements_.front().timeStamp > eraseUntil)
            return 0;

        okvis::ImuMeasurementDeque::iterator eraseEnd;
        int removed = 0;
        for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
            eraseEnd = it;
            if (it->timeStamp >= eraseUntil)
                break;
            ++removed;
        }

        imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

        return removed;
    }



}