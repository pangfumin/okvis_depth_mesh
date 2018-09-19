/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nichoilas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file epipolar_geometry.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:17:13 (Fri)
 */


 // todo: adapt to different camera intrinsic
#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>

#include "flame/types.h"
#include "flame/utils/assert.h"
#include "flame/utils/image_utils.h"

namespace flame {

namespace stereo {

/**
 * \brief Class that represents an epipolar geometry setup.
 *
 * Useful for epipolar geometry queries (epipolar lines, projections, depth,
 * etc.)
 */
template <typename Scalar>
class EpipolarGeometry final {
  // Convenience aliases.
  using Point2s = cv::Point_<Scalar>; // TODO(wng): Use core types.
  using Vector3s = Vector3<Scalar>;
  using Matrix3s = Matrix3<Scalar>;
  using Quaternions = Quaternion<Scalar>;

 public:
  /**
   * \brief Constructor.
   *
   * The comparison camera is the camera that pixels are projected onto to form
   * epipolar lines and compute disparity. The reference camera is the camera
   * that depths are define relative to.
   *
   * @param[in] K Camera intrinsic matrix.
   * @param[in] Kinv Inverse camera intrinsic matrix.
   */
    EpipolarGeometry(const Matrix3s& KA, const Matrix3s& KAinv,
                     const Matrix3s& KB, const Matrix3s& KBinv) :
            KA_(KA),
            KAinv_(KAinv),
            KB_(KB),
            KBinv_(KBinv),
            q_B_A_(),
            t_B_A_(),
            t_A_B_(),
            KB_R_KAinv3_(),
            KBt_(),
            epipole_() {}

  EpipolarGeometry() = default;
  ~EpipolarGeometry() = default;

  EpipolarGeometry(const EpipolarGeometry& rhs) = default;
  EpipolarGeometry& operator=(const EpipolarGeometry& rhs) = default;

  EpipolarGeometry(EpipolarGeometry&& rhs) = default;
  EpipolarGeometry& operator=(EpipolarGeometry&& rhs) = default;

  /**
   * \brief Load an epipolar geometry setup.
   *
   * @param[in] q_B_A
   * @param[in] t_B_A
   */
  void loadGeometry(const Quaternions& q_B_A,
                    const Vector3s& t_B_A) {
    q_B_A_ = q_B_A;
    t_B_A_ = t_B_A;
    t_A_B_ = -(q_B_A_.inverse() * t_B_A_);
    KB_R_KAinv_ = KB_ * q_B_A_.toRotationMatrix() * KAinv_;
    KB_R_KAinv3_ = KB_R_KAinv_.row(2);
    KBt_ = KB_ * t_B_A_;


    // If B frame is in front of A frame.
    if (t_B_A_(2) > 0) {
      // Precompute epipole.
      epipole_.x = (KB_(0, 0) * t_B_A_(0) + KB_(0, 2) * t_B_A_(2)) /
              t_B_A_(2);
      epipole_.y = (KB_(1, 1) * t_B_A_(1) + KB_(1, 2) * t_B_A_(2)) /
              t_B_A_(2);
    }
    return;
  }

  /**
   * \brief Helper function for perspective projection.
   *
   * @param[in] K Camera intrinsic matrix.
   * @param[in] q Orientation of camera in world.
   * @param[in] t Translation of camera in world.
   * @param[in] p Point to project in world.
   * @return Projected pixel.
   */
  static Point2s project(const Matrix3s& K, const Quaternions& q,
                         const Vector3s& t, const Vector3s& p) {
    Vector3s p_cam(q.inverse() * (p - t));
    return Point2s((K(0, 0) * p_cam(0) + K(0, 2) * p_cam(2)) / p_cam(2),
                   (K(1, 1) * p_cam(1) + K(1, 2) * p_cam(2)) / p_cam(2));
  }

  /**
   * \brief Project pixel u_ref into comparison frame assuming inverse depth.
   *
   * @param[in] u_ref Reference pixel.
   * @param[in] idepth Inverse depth.
   */
  Point2s project(const Point2s& u_A, Scalar idepth) const {
    FLAME_ASSERT(idepth >= 0.0f);
    if (idepth == 0.0f) {
      Point2s u_max;
      maxDepthProjection(u_A, &u_max);
      return u_max;
    }

    Scalar depth = 1.0f / idepth;
    Vector3s u_A_hom(u_A.x * depth, u_A.y * depth, depth);
    Vector3s u_B_hom(KB_R_KAinv_ * u_A_hom + KBt_);

    FLAME_ASSERT(utils::fast_abs(u_B_hom(2)) > 0.0f);

    Scalar inv_u_B_hom2 = 1.0f / u_B_hom(2);
    return Point2s(u_B_hom(0) * inv_u_B_hom2, u_B_hom(1) * inv_u_B_hom2);
  }

  /**
   * \brief Project pixel u_ref into comparison frame assuming inverse depth.
   *
   * @param[in] u_ref Reference pixel.
   * @param[in] idepth Inverse depth.
   * @param[out] u_cmp Projected pixel.
   * @param[out] new_idepth New inverse depth.
   */
  void project(const Point2s& u_A, Scalar idepth,
               Point2s* u_B, Scalar* new_idepth) const {
    FLAME_ASSERT(idepth >= 0.0f);
    if (idepth == 0.0f) {
      Point2s u_max;
      maxDepthProjection(u_A, &u_max);
      *u_B = u_max;
      *new_idepth = 0.0f;
      return;
    }

    Scalar depth = 1.0f / idepth;
    Vector3s p_A(KAinv_(0, 0) * u_A.x + KAinv_(0, 2),
                   KAinv_(1, 1) * u_A.y + KAinv_(1, 2),
                   1.0f);
    p_A *= depth;

    Vector3s p_B(q_B_A_ * p_A + t_B_A_);
    Vector3s u_B3(KB_(0, 0) * p_B(0) + KB_(0, 2) * p_B(2),
                    KB_(1, 1) * p_B(1) + KB_(1, 2) * p_B(2),
                  p_B(2));
    FLAME_ASSERT(fabs(u_B3(2)) > 0.0f);

    *new_idepth = 1.0f / p_B(2);
    u_B->x = u_B3(0) * (*new_idepth);
    u_B->y = u_B3(1) * (*new_idepth);
    return;
  }

  /**
   * \brief Compute projection of pixel with infinite depth.
   *
   * Compute the projection of pixel u_ref into the comparison image assuming
   * infinite depth.
   *
   * @param u_ref[in] Pixel in the reference image to project.
   * @param u_inf[out] The projection corresponding to infinite depth.
   */
  void maxDepthProjection(const Point2s& u_A, Point2s* u_inf) const {
    Vector3s u_A_hom(u_A.x, u_A.y, 1.0f);
    Vector3s u_B_hom(KB_R_KAinv_ * u_A_hom);

    FLAME_ASSERT(fabs(u_B_hom(2)) > 0.0f);

    Scalar inv_u_B_hom2 = 1.0f / u_B_hom(2);
    u_inf->x = u_B_hom(0) * inv_u_B_hom2;
    u_inf->y = u_B_hom(1) * inv_u_B_hom2;
    return;
  }

  /**
   * \brief Compute the epiline endpoint.
   *
   * There are several ways to compute the epiline endpoint. The most natural
   * way is if the A camera is in front of the B camera in the
   * comparison camera frame (that is t_B_A_z > 0). Then the epiline
   * endpoint is just the epipole (i.e. the projection of the reference camera
   * into the comparison camera). This corresponds to the point in the world
   * having 0 depth.
   *
   * Things get more complicated however if t_ref_to_cmp_z <= 0 (i.e. the
   * reference camera lies at the same z or behind the comparison camera).

   * If t_ref_to_cmp_z = 0, then the epipole lies at infinity and all epilines
   * are parallel (typically the case for a traditional stereo setup). In this
   * case, the vector (fx * t_ref_to_cmp_x, fy * t_ref_to_cmp_y) is parallel to
   * the epiline. Given the infinity point, the minimum depth point is simply
   * the infinite point plus a large value times this vector.
   *
   * If t_ref_to_cmp_z < 0, then the minimum possible depth of the point must be
   * 0 in order to be projected into both camera. In this case, we simply
   * compute the depth in the reference frame such * that the point has depth 1
   * in the comparison frame and then project this * point into the comparison
   * camera
   *
   * It is also possible to compute the epiline using the fundamental matrix
   * F. If epiline is parameterized by the implicit equation l^T u_cmp = 0,
   * where u_cmp are homogenous pixels in the comparison image, then l = F
   * u_ref. This formulation, however, does not give the *direction* of the
   * epiline from far depth to near depth, which is what we would like.
   *
   * @param u_ref[in] Pixel in the reference image to project.
   * @param u_min[out] The projection corresponding to minimum depth.
   */
  void minDepthProjection(const Point2s& u_A, Point2s* u_min) const {
    if (t_B_A_(2) > 0) {
      *u_min = epipole_;
    } else if (t_B_A_(2) == 0) {
      // Compute epiline direction.
      Point2s epi(KB_(0, 0) * t_B_A_(0), KB_(1, 1) * t_B_A_(1));
      Point2s u_inf;
      maxDepthProjection(u_A, &u_inf);
      *u_min = u_inf + 1e6 * epi;
    } else {
      // Compute depth in the ref frame such that point has depth 1 in comparison
      // frame.
      Vector3s qp_A(KAinv_(0, 0) * u_A.x + KAinv_(0, 2),
                      KAinv_(1, 1) * u_A.y + KAinv_(1, 2),
                      1.0f);
      Vector3s qp_B = q_B_A_ * qp_A;
      Scalar min_depth = (1.0f - t_B_A_(2)) / qp_B(2);

      Vector3s p_B(min_depth * qp_B + t_B_A_);
      FLAME_ASSERT(p_B(2) > 0.0f);

      u_min->x = (KB_(0, 0) * p_B(0) + KB_(0, 2) * p_B(2)) / p_B(2);
      u_min->y = (KB_(1, 1) * p_B(1) + KB_(1, 2) * p_B(2)) / p_B(2);
    }

    return;
  }

  /**
   * \brief Compute epipolar line corresponding to pixel.
   *
   * Computes the epipolar line of pixel u_ref in the cmp image. The line points
   * from infinite depth to minimum depth and is computed by finding the pixels
   * corresponding to infinite depth and minimum depth in the comparison image.
   *
   * It is also possible to compute the epiline using the fundamental matrix
   * F. If epiline is parameterized by the implicit equation l^T u_cmp = 0,
   * where u_cmp are homogenous pixels in the comparison image, then l = F
   * u_ref. This formulation, however, does not give the *direction* of the
   * epiline from far depth to near depth, which is what we would like.
   *
   * @param u_ref[in] Reference pixel.
   * @param u_inf[out] Start of epipolar line (point of infinite depth).
   & @param epi[out] Epipolar unit vector.
  */
  void epiline(const Point2s& u_A, Point2s* u_inf, Point2s* epi) const {
    Point2s u_zero;
    minDepthProjection(u_A, &u_zero);
    maxDepthProjection(u_A, u_inf);
    *epi = u_zero - *u_inf;
    Scalar norm2 = epi->x*epi->x + epi->y*epi->y;

    if (norm2 > 1e-10) {
      Scalar inv_norm = 1.0f / sqrt(norm2);
      epi->x *= inv_norm;
      epi->y *= inv_norm;
    } else {
      // If u_zero == u_inf, then epi mag is 0.
      epi->x = 0.0f;
      epi->y = 0.0f;
    }

    return;
  }

  /**
   * @brief Return the epiline that corresponds to u_ref in the reference
   * image. This is the projection of the epipolar plane onto the reference
   * image at u_ref. It points from near depth to far depth (opposite of what's
   * returned from epiline).
   *
   * @param[in] u_ref Reference pixel.
   * @param[out] epi Epipolar line.
   */
  void referenceEpiline(const Point2s& u_A, Point2s* epi) const {
    // Get epiline in reference image for the template.
    // calculate the plane spanned by the two camera centers and the point (x,y,1)
    // intersect it with the keyframe's image plane (at depth = 1)
    // This is the epipolar line in the keyframe.
    Point2s epi_A;
    epi_A.x = -KA_(0, 0) * t_A_B_(0) +
            t_A_B_(2)*(u_A.x - KA_(0, 2));
    epi_A.y = -KA_(1, 1) * t_A_B_(1) +
            t_A_B_(2)*(u_A.y - KA_(1, 2));

    Scalar epi_A_norm2 = epi_A.x * epi_A.x + epi_A.y * epi_A.y;
    FLAME_ASSERT(epi_A_norm2 > 0);
    Scalar inv_epi_A_norm = 1.0f / sqrt(epi_A_norm2);
    epi_A.x *= inv_epi_A_norm;
    epi_A.y *= inv_epi_A_norm;

    *epi = epi_A;

    return;
  }

  /**
   * \brief Compute disparity from pixel correspondence.
   *
   * @param u_ref[in] Reference pixel.
   * @param u_cmp[in] Comparison pixel.
   * @param u_inf[out] Endpoint of epipolar line (point of infinite depth).
   * @param epi[out] Epipolar line direction.
   * @param disparity[out] Disparity.
   */
  Scalar disparity(const Point2s& u_A, const Point2s& u_B,
                   Point2s* u_inf, Point2s* epi) const {
    epiline(u_A, u_inf, epi);
    return epi->x*(u_B.x - u_inf->x) + epi->y*(u_B.y - u_inf->y);
  }
  Scalar disparity(const Point2s& u_A, const Point2s& u_B) const {
    Point2s u_inf, epi;
    return disparity(u_A, u_B, &u_inf, &epi);
  }
  Scalar disparity(const Point2s& u_A, const Point2s& u_B,
                   const Point2s& u_inf, const Point2s& epi) const {
    return epi.x*(u_B.x - u_inf.x) + epi.y*(u_B.y - u_inf.y);
  }

  /**
   * \brief Compute depth from disparity.
   *
   * @param u_ref[in] Reference pixel
   * @param u_inf[in] Epipolar line start point (infinite depth).
   * @param epi[in] Epipolar line direction.
   * @param disparity[in] Disparity.
   * @return depth
   */
  Scalar disparityToDepth(const Point2s& u_A, const Point2s& u_inf,
                          const Point2s& epi, const Scalar disparity) const {
    FLAME_ASSERT(disparity >= 0.0f);
    Scalar w = KB_R_KAinv3_(0) * u_A.x + KB_R_KAinv3_(1) * u_A.y + KB_R_KAinv3_(2);
    Point2s A(w * disparity * epi);
    Point2s b(KBt_(0) - KBt_(2)*(u_inf.x + disparity * epi.x),
              KBt_(1) - KBt_(2)*(u_inf.y + disparity * epi.y));

    Scalar ATA = A.x*A.x + A.y*A.y;
    Scalar ATb = A.x*b.x + A.y*b.y;

    FLAME_ASSERT(ATA > 0.0f);

    return ATb/ATA;
  }

  /**
   * \brief Compute inverse depth from disparity.
   *
   * @param KR_ref_to_cmpKinv3 Third row of K*R_ref_to_cmp*Kinv.
   * @param Kt_ref_to_cmp[in] K * translation from ref to cmp.
   * @param u_ref[in] Reference pixel
   * @param u_inf[in] Epipolar line start point (infinite depth).
   * @param epi[in] Epipolar line direction.
   * @param disparity[in] Disparity.
   * @return inverse depth
   */
  Scalar disparityToInverseDepth(const Point2s& u_A, const Point2s& u_inf,
                                 const Point2s& epi,
                                 const Scalar disparity) const {
    FLAME_ASSERT(disparity >= 0.0f);
    Scalar w = KB_R_KAinv3_(0) * u_A.x + KB_R_KAinv3_(1) * u_A.y + KB_R_KAinv3_(2);
    Point2s A(KBt_(0) - KBt_(2)*(u_inf.x + disparity * epi.x),
              KBt_(1) - KBt_(2)*(u_inf.y + disparity * epi.y));
    Point2s b(w * disparity * epi);

    Scalar ATA = A.x*A.x + A.y*A.y;
    Scalar ATb = A.x*b.x + A.y*b.y;

    FLAME_ASSERT(ATA > 0.0f);

    return ATb/ATA;
  }



 private:
  // Camera parameters.
  Matrix3s KA_;
  Matrix3s KAinv_;
    Matrix3s KB_;
    Matrix3s KBinv_;

  // Geometry.
  Quaternions q_B_A_;
  Vector3s t_B_A_;
  Vector3s t_A_B_;
  Matrix3s KB_R_KAinv_;
  Vector3s KB_R_KAinv3_;
  Vector3s KBt_;
  Point2s epipole_; // Projection of cmp camera in ref camera.
};

}  // namespace stereo

}  // namespace flame
