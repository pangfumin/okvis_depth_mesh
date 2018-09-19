#ifndef _ESTIMATE_PIPELINE_H_
#define _ESTIMATE_PIPELINE_H_

#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Frontend.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/VioInterface.hpp>

#include <flame/params.h>

namespace flame {
    class MeshEstimator;
}

namespace okvis {
    class EstimatePipeline: public VioInterface {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

        EstimatePipeline(okvis::VioParameters& parameters,
                         flame::Params mesh_parameter = flame::Params());

        /// \brief Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
        virtual ~EstimatePipeline();

        /// \name Add measurements to the algorithm
        /// \{
        /**
         * \brief              Add a new image.
         * \param stamp        The image timestamp.
         * \param cameraIndex  The index of the camera that the image originates from.
         * \param image        The image.
         * \param keypoints    Optionally aready pass keypoints. This will skip the detection part.
         * \param asKeyframe   Use the new image as keyframe. Not implemented.
         * \warning The frame consumer loop does not support using existing keypoints yet.
         * \warning Already specifying whether this frame should be a keyframe is not implemented yet.
         * \return             Returns true normally. False, if the previous one has not been processed yet.
         */
        virtual bool addImage(const okvis::Time & stamp, size_t cameraIndex,
                              const cv::Mat & image,
                              const std::vector<cv::KeyPoint> * keypoints = 0,
                              bool* asKeyframe = 0){
            OKVIS_THROW(Exception, "not implemented");
        }

        bool addImage(const okvis::Time & stamp, const cv::Mat & image0,
                              const cv::Mat & image1,
                              const std::vector<cv::KeyPoint> * keypoints = 0,
                              bool* asKeyframe = 0);

        /**
       * \brief             Add an abstracted image observation.
       * \param stamp       The timestamp for the start of integration time for the image.
       * \param cameraIndex The index of the camera.
       * \param keypoints   A vector where each entry represents a [u,v] keypoint measurement. Also set the size field.
       * \param landmarkIds A vector of landmark ids for each keypoint measurement.
       * \param descriptors A matrix containing the descriptors for each keypoint.
       * \param asKeyframe  Optionally force keyframe or not.
       * \return            Returns true normally. False, if the previous one has not been processed yet.
       */
        virtual bool addKeypoints(const okvis::Time & stamp, size_t cameraIndex,
                                  const std::vector<cv::KeyPoint> & keypoints,
                                  const std::vector<uint64_t> & landmarkIds,
                                  const cv::Mat& descriptors = cv::Mat(),
                                  bool* asKeyframe = 0)  {
            OKVIS_THROW(Exception, "not implemented");
        }

        /**
          * \brief          Add an IMU measurement.
          * \param stamp    The measurement timestamp.
          * \param alpha    The acceleration measured at this time.
          * \param omega    The angular velocity measured at this time.
          * \return Returns true normally. False if the previous one has not been processed yet.
          */
        virtual bool addImuMeasurement(const okvis::Time & stamp,
                                       const Eigen::Vector3d & alpha,
                                       const Eigen::Vector3d & omega);

        /// \brief                      Add a position measurement.
        /// \warning Not Implemented.
        /*
        /// \param stamp                The measurement timestamp
        /// \param position             The position in world frame
        /// \param positionCovariance   The position measurement covariance matrix.
        */
        virtual void addPositionMeasurement(
                const okvis::Time & /*stamp*/, const Eigen::Vector3d & /*position*/,
                const Eigen::Vector3d & /*positionOffset*/,
                const Eigen::Matrix3d & /*positionCovariance*/) {
            OKVIS_THROW(Exception, "not implemented");
        }

        /// \brief                       Add a position measurement.
        /// \warning Not Implemented.
        /*
        /// \param stamp                 The measurement timestamp
        /// \param lat_wgs84_deg         WGS84 latitude [deg]
        /// \param lon_wgs84_deg         WGS84 longitude [deg]
        /// \param alt_wgs84_deg         WGS84 altitude [m]
        /// \param positionOffset        Body frame antenna position offset [m]
        /// \param positionCovarianceENU The position measurement covariance matrix.
        */
        virtual void addGpsMeasurement(
                const okvis::Time & /*stamp*/, double /*lat_wgs84_deg*/,
                double /*lon_wgs84_deg*/, double /*alt_wgs84_deg*/,
                const Eigen::Vector3d & /*positionOffset*/,
                const Eigen::Matrix3d & /*positionCovarianceENU*/) {
            OKVIS_THROW(Exception, "not implemented");
        }

        /// \brief                      Add a magnetometer measurement.
        /// \warning Not Implemented.
        /*
        /// \param stamp                The measurement timestamp
        /// \param fluxDensityMeas      Measured magnetic flux density (sensor frame) [uT]
        /// \param stdev                Measurement std deviation [uT]
        */
        /// \return                     Returns true normally. False, if the previous one has not been processed yet.
        virtual void addMagnetometerMeasurement(
                const okvis::Time & /*stamp*/,
                const Eigen::Vector3d & /*fluxDensityMeas*/, double /*stdev*/) {
            OKVIS_THROW(Exception, "not implemented");
        }

        /// \brief                      Add a static pressure measurement.
        /// \warning Not Implemented.
        /*
        /// \param stamp                The measurement timestamp
        /// \param staticPressure       Measured static pressure [Pa]
        /// \param stdev                Measurement std deviation [Pa]
        */
        virtual void addBarometerMeasurement(const okvis::Time & /*stamp*/,
                                             double /*staticPressure*/,
                                             double /*stdev*/) {
            OKVIS_THROW(Exception, "not implemented");
        }

        /// \brief                      Add a differential pressure measurement.
        /// \warning Not Implemented.
        /*
        /// \param stamp                The measurement timestamp
        /// \param differentialPressure Measured differential pressure [Pa]
        /// \param stdev                Measurement std deviation [Pa]
        */
        virtual void addDifferentialPressureMeasurement(
                const okvis::Time & /*stamp*/, double /*differentialPressure*/,
                double /*stdev*/) {
            OKVIS_THROW(Exception, "not implemented");
        }

        /**
         * @brief This is just handy for the python interface.
         * @param stamp       The image timestamp
         * @param cameraIndex The index of the camera that the image originates from.
         * @param image       The image.
         * @return Returns true normally. False, if the previous one has not been processed yet.
         */
        bool addEigenImage(const okvis::Time & stamp, size_t cameraIndex,
                           const EigenImage & image) {
            OKVIS_THROW(Exception, "not implemented");
        }


        /**
         * \brief Set the blocking variable that indicates whether the addMeasurement() functions
         *        should return immediately (blocking=false), or only when the processing is complete.
         */
        virtual void setBlocking(bool blocking) {
            OKVIS_THROW(Exception, "not implemented");

        }


        void display();

        /**
       * @brief Get a subset of the recorded IMU measurements.
       * @param start The first IMU measurement in the return value will be older than this timestamp.
       * @param end The last IMU measurement in the return value will be newer than this timestamp.
       * @remark This function is threadsafe.
       * @return The IMU Measurement spanning at least the time between start and end.
       */
        okvis::ImuMeasurementDeque getImuMeasurments(okvis::Time& start,
                                                     okvis::Time& end);

        /**
         * @brief Remove IMU measurements from the internal buffer.
         * @param eraseUntil Remove all measurements that are strictly older than this time.
         * @return The number of IMU measurements that have been removed
         */
        int deleteImuMeasurements(const okvis::Time& eraseUntil);


    private:
        void init();


        struct OptimizationResults {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            okvis::Time stamp;                          ///< Timestamp of the optimized/propagated pose.
            okvis::kinematics::Transformation T_WS;     ///< The pose.
            okvis::SpeedAndBias speedAndBiases;         ///< The speeds and biases.
            Eigen::Matrix<double, 3, 1> omega_S;        ///< The rotational speed of the sensor.
            /// The relative transformation of the cameras to the sensor (IMU) frame
            std::vector<okvis::kinematics::Transformation,
                    Eigen::aligned_allocator<okvis::kinematics::Transformation> > vector_of_T_SCi;
            okvis::MapPointVector landmarksVector;      ///< Vector containing the current landmarks.
            okvis::MapPointVector transferredLandmarks; ///< Vector of the landmarks that have been marginalized out.
            bool onlyPublishLandmarks;                  ///< Boolean to signalise the publisherLoop() that only the landmarks should be published
        };

        /// @name State variables
        /// @{

        okvis::SpeedAndBias speedAndBiases_propagated_;     ///< The speeds and IMU biases propagated by the IMU measurements.
        /// \brief The IMU parameters.
        /// \warning Duplicate of parameters_.imu
        okvis::ImuParameters imu_params_;
        okvis::kinematics::Transformation T_WS_propagated_; ///< The pose propagated by the IMU measurements
        std::shared_ptr<okvis::MapPointVector> map_;        ///< The map. Unused.

        // lock lastState_mutex_ when accessing these
        /// \brief Resulting pose of the last optimization
        /// \warning Lock lastState_mutex_.
        okvis::kinematics::Transformation lastOptimized_T_WS_;
        /// \brief Resulting speeds and IMU biases after last optimization.
        /// \warning Lock lastState_mutex_.
        okvis::SpeedAndBias lastOptimizedSpeedAndBiases_;
        /// \brief Timestamp of newest frame used in the last optimization.
        /// \warning Lock lastState_mutex_.
        okvis::Time lastOptimizedStateTimestamp_;
        /// This is set to true after optimization to signal the IMU consumer loop to repropagate
        /// the state from the lastOptimizedStateTimestamp_.
        std::atomic_bool repropagationNeeded_;

        /// @}

        okvis::Time lastAddedStateTimestamp_; ///< Timestamp of the newest state in the Estimator.
        okvis::Time lastAddedImageTimestamp_; ///< Timestamp of the newest image added to the image input queue.


        /// @name Measurement input queues

        /// \brief The IMU measurements.
        /// \warning Lock with imuMeasurements_mutex_.
        okvis::ImuMeasurementDeque imuMeasurements_;

        /// @}
        /// @name Algorithm objects.
        /// @{

#ifdef USE_MOCK
        okvis::MockVioBackendInterface& estimator_;
  okvis::MockVioFrontendInterface& frontend_;
#else
        okvis::Estimator estimator_;    ///< The backend estimator.
        okvis::Frontend frontend_;      ///< The frontend.
#endif

        /// @}

        size_t numCameras_;     ///< Number of cameras in the system.
        size_t numCameraPairs_; ///< Number of camera pairs in the system.

        okvis::VioParameters parameters_; ///< The parameters and settings.

        std::vector<cv::Mat>  displayImages_;
        /*
         *  Mesh estimate
         *
         */

        flame::Params meshParams;
        std::shared_ptr<flame::MeshEstimator> meshEstimatorPtr_;
    };
}

#endif