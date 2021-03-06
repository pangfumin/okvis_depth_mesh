

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <atomic>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include <okvis/VioParametersReader.hpp>
#include <okvis/estimate_pipeline.hpp>

#include <boost/filesystem.hpp>

class PoseViewer
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 500.0;
  PoseViewer()
  {
    cv::namedWindow("OKVIS Top View");
    _image.create(imageSize, imageSize, CV_8UC3);
    drawing_ = false;
    showing_ = false;
  }
  // this we can register as a callback
  void publishFullStateAsCallback(
      const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS,
      const Eigen::Matrix<double, 9, 1> & speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
  {

    // just append the path
    Eigen::Vector3d r = T_WS.r();
    Eigen::Matrix3d C = T_WS.C();
    _path.push_back(cv::Point2d(r[0], r[1]));
    _heights.push_back(r[2]);
    // maintain scaling
    if (r[0] - _frameScale < _min_x)
      _min_x = r[0] - _frameScale;
    if (r[1] - _frameScale < _min_y)
      _min_y = r[1] - _frameScale;
    if (r[2] < _min_z)
      _min_z = r[2];
    if (r[0] + _frameScale > _max_x)
      _max_x = r[0] + _frameScale;
    if (r[1] + _frameScale > _max_y)
      _max_y = r[1] + _frameScale;
    if (r[2] > _max_z)
      _max_z = r[2];
    _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

    // draw it
    while (showing_) {
    }
    drawing_ = true;
    // erase
    _image.setTo(cv::Scalar(10, 10, 10));
    drawPath();
    // draw axes
    Eigen::Vector3d e_x = C.col(0);
    Eigen::Vector3d e_y = C.col(1);
    Eigen::Vector3d e_z = C.col(2);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
        cv::Scalar(0, 0, 255), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
        cv::Scalar(0, 255, 0), 1, CV_AA);
    cv::line(
        _image,
        convertToImageCoordinates(_path.back()),
        convertToImageCoordinates(
            _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
        cv::Scalar(255, 0, 0), 1, CV_AA);

    // some text:
    std::stringstream postext;
    postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
    cv::putText(_image, postext.str(), cv::Point(15,15),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    std::stringstream veltext;
    veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
    cv::putText(_image, veltext.str(), cv::Point(15,35),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

    drawing_ = false; // notify
  }
  void display()
  {
    while (drawing_) {
    }
    showing_ = true;
    cv::imshow("OKVIS Top View", _image);
    showing_ = false;
    cv::waitKey(1);
  }
 private:
  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const
  {
    cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
    return cv::Point2d(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
  }
  void drawPath()
  {
    for (size_t i = 0; i + 1 < _path.size(); ) {
      cv::Point2d p0 = convertToImageCoordinates(_path[i]);
      cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
      cv::Point2d diff = p1-p0;
      if(diff.dot(diff)<2.0){
        _path.erase(_path.begin() + i + 1);  // clean short segment
        _heights.erase(_heights.begin() + i + 1);
        continue;
      }
      double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z)
                      * 0.5 / (_max_z - _min_z);
      cv::line(
          _image,
          p0,
          p1,
          rel_height * cv::Scalar(255, 0, 0)
              + (1.0 - rel_height) * cv::Scalar(0, 0, 255),
          1, CV_AA);
      i++;
    }
  }
  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -0.5;
  double _min_y = -0.5;
  double _min_z = -0.5;
  double _max_x = 0.5;
  double _max_y = 0.5;
  double _max_z = 0.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;
  std::atomic_bool showing_;
};

// this is just a workbench. most of the stuff here will go into the Frontend class.
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  if (argc != 3 && argc != 4) {
    LOG(ERROR)<<
    "Usage: ./" << argv[0] << " configuration-yaml-file dataset-folder [skip-first-seconds]";
    return -1;
  }

  okvis::Duration deltaT(0.0);
  if (argc == 4) {
    deltaT = okvis::Duration(atof(argv[3]));
  }

  // read configuration file
  std::string configFilename(argv[1]);

  okvis::VioParametersReader vio_parameters_reader(configFilename);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);

  okvis::EstimatePipeline okvis_estimator(parameters);

  PoseViewer poseViewer;
  okvis_estimator.setFullStateCallback(
      std::bind(&PoseViewer::publishFullStateAsCallback, &poseViewer,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4));
//
//  okvis_estimator.setBlocking(true);

  // the folder path
  std::string path(argv[2]);

  const unsigned int numCameras = parameters.nCameraSystem.numCameras();

  // open the IMU file
  std::string line;
  std::ifstream imu_file(path + "/imu0/data.csv");
  if (!imu_file.good()) {
    LOG(ERROR)<< "no imu file found at " << path+"/imu0/data.csv";
    return -1;
  }
  int number_of_lines = 0;
  while (std::getline(imu_file, line))
    ++number_of_lines;
  LOG(INFO)<< "No. IMU measurements: " << number_of_lines-1;
  if (number_of_lines - 1 <= 0) {
    LOG(ERROR)<< "no imu messages present in " << path+"/imu0/data.csv";
    return -1;
  }
  // set reading position to second line
  imu_file.clear();
  imu_file.seekg(0, std::ios::beg);
  std::getline(imu_file, line);

  std::vector<okvis::Time> times;
  okvis::Time latest(0);
  int num_camera_images = 0;
  std::vector < std::vector < std::string >> image_names(numCameras);
  for (size_t i = 0; i < numCameras; ++i) {
    num_camera_images = 0;
    std::string folder(path + "/cam" + std::to_string(i) + "/data");

    for (auto it = boost::filesystem::directory_iterator(folder);
        it != boost::filesystem::directory_iterator(); it++) {
      if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
        num_camera_images++;
        image_names.at(i).push_back(it->path().filename().string());
      } else {
        continue;
      }
    }

    if (num_camera_images == 0) {
      LOG(ERROR)<< "no images at " << folder;
      return 1;
    }

    LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
    // the filenames are not going to be sorted. So do this here
    std::sort(image_names.at(i).begin(), image_names.at(i).end());
  }

  std::vector < std::vector < std::string > ::iterator
      > cam_iterators(numCameras);
  for (size_t i = 0; i < numCameras; ++i) {
    cam_iterators.at(i) = image_names.at(i).begin();
  }

  int counter = 0;
  okvis::Time start(0.0);
  while (true) {

    // check if at the end
    for (size_t i = 0; i < numCameras; ++i) {
      if (cam_iterators[i] == image_names[i].end()) {
        std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
        cv::waitKey();
        return 0;
      }
    }

    /// add images
    okvis::Time t;

    cv::Mat filtered0 = cv::imread(
          path + "/cam" + std::to_string(0) + "/data/" + *cam_iterators.at(0),
          cv::IMREAD_GRAYSCALE);
    cv::Mat filtered1 = cv::imread(
            path + "/cam" + std::to_string(1) + "/data/" + *cam_iterators.at(1),
            cv::IMREAD_GRAYSCALE);


      std::string nanoseconds = cam_iterators.at(0)->substr(
          cam_iterators.at(0)->size() - 13, 9);
      std::string seconds = cam_iterators.at(0)->substr(
          0, cam_iterators.at(0)->size() - 13);
      t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
      if (start == okvis::Time(0.0)) {
        start = t;
      }

      // get all IMU measurements till then
      okvis::Time t_imu = start;
      do {
        if (!std::getline(imu_file, line)) {
          std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
          cv::waitKey();
          return 0;
        }

        std::stringstream stream(line);
        std::string s;
        std::getline(stream, s, ',');
        std::string nanoseconds = s.substr(s.size() - 9, 9);
        std::string seconds = s.substr(0, s.size() - 9);

        Eigen::Vector3d gyr;
        for (int j = 0; j < 3; ++j) {
          std::getline(stream, s, ',');
          gyr[j] = std::stof(s);
        }

        Eigen::Vector3d acc;
        for (int j = 0; j < 3; ++j) {
          std::getline(stream, s, ',');
          acc[j] = std::stof(s);
        }

        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

        // add the IMU measurement for (blocking) processing
        if (t_imu - start + okvis::Duration(1.0) > deltaT) {
          okvis_estimator.addImuMeasurement(t_imu, acc, gyr);
        }

      } while (t_imu <= t);

      // add the image to the frontend for (blocking) processing
      if (t - start > deltaT) {
        okvis_estimator.addImage(t,  filtered0, filtered1);
      }

      cam_iterators[0]++;
      cam_iterators[1]++;

    ++counter;

    poseViewer.display();

    // display progress
    if (counter % 20 == 0) {
      std::cout << "\rProgress: "
          << int(double(counter) / double(num_camera_images) * 100) << "%  "
          << std::flush;
    }

  }

  std::cout << std::endl << std::flush;
  return 0;
}
