/**
 * @file AmazonBucketAPI.h.
 *
 * @brief
 */

#ifndef AMAZON_BUCKET_API_H
#define AMAZON_BUCKET_API_H

/**
 * @namespace Networking
 *
 * @brief
 */
namespace Networking
{

/**
 * @class AmazonBucketAPI
 *
 * @brief
 */
class AmazonBucketAPI
{

public:

    /**
     * @brief
     */
    AmazonBucketAPI();

    /**
     * @brief
     */
    ~AmazonBucketAPI();

private:

    /**
     * @brief
     */
    static void InitializeAwsSdk();

    ///
    inline static bool isAwsSdkInitialized = false;

};

}

#endif // AMAZON_BUCKET_API_H
