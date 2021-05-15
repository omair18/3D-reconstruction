/**
 * @file KeyPoint.h.
 *
 * @brief
 */

#ifndef KEYPOINT_H
#define KEYPOINT_H

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

/**
 * @struct KeyPoint
 *
 * @brief
 */
struct KeyPoint final
{
    /// X-coordinate of the keypoint.
    float x;

    /// Y-coordinate of the keypoint.
    float y;

    /// Diameter of the meaningful keypoint neighborhood.
    float size;

    /// Computed orientation of the keypoint (-1 if not applicable). It's in [0,360) degrees and measured relative to
    /// image coordinate system in clockwise.
    float angle;

    /// Response by which the most strong keypoints have been selected. Can be used for the further sorting or
    /// subsampling
    float response;

    /// Octave (pyramid layer) from which the keypoint has been extracted
    int octave;

    /// Object class (if the keypoints need to be clustered by an object they belong to)
    int classId;
};

}

#endif // KEYPOINT_H
