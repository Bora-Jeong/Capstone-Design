#include <jni.h>
#include <vector>
#include <set>
#include "KalmanTracker.h"
#include "HungarianAlgorithm.h"
#include <android/log.h>
using namespace std;



#define LOGD(...) __android_log_print(_DEBUG,LOG_TAG,__VA_ARGS__)


extern "C" JNIEXPORT jstring
Java_org_tensorflow_lite_examples_detection_CameraActivity_stringFromJNI(JNIEnv *env, jobject /* this */) {
    string  s;
    s += "HI";
    return env->NewStringUTF(s.c_str());
}


typedef struct TrackingBox
{
    float confidence;
    int id;
    Rect_<float> box;
    int title;
}TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}


//__android_log_print(ANDROID_LOG_DEBUG, "JNI", "inCArray : %d", length);


int frame_count = 0;
int max_age = 1;
int min_hits = 3;
double iouThreshold = 0.3;
vector<KalmanTracker> trackers;

extern "C" JNIEXPORT jfloatArray
Java_org_tensorflow_lite_examples_detection_DetectorActivity_sort(JNIEnv *env, jobject /* this */ , jfloatArray inJNIArray) {

    // Step 1: Convert the incoming JNI jintarray to C's jfloat[]
    jfloat *inCArray = env->GetFloatArrayElements(inJNIArray, 0);
    if (NULL == inCArray) return NULL;
    jsize length = env->GetArrayLength(inJNIArray);


    // Step 2 : Perform its intended operations

    // 1. read detection file
    vector<TrackingBox> detData;
    float px, py, w, h;
    for(int i = 0; i < length / 7; i++){
        TrackingBox tb;
        tb.confidence = inCArray[i * 7 + 0];
        tb.id = (int)inCArray[i * 7 + 1];
        px = inCArray[i * 7 + 2];
        py = inCArray[i * 7  + 3];
        w = inCArray[i * 7 + 4];
        h = inCArray[i * 7 + 5];
        tb.title = (int)inCArray[i * 7 + 6];
        tb.box = Rect_<float>(px, py, w, h);
        detData.push_back(tb);
    }


    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;


    ///////////////////////////////////////
    // main
    frame_count++;


    if (trackers.size() == 0) // the first frame met
    {
        KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

        // initialize kalman trackers using first detections.
        for (unsigned int i = 0; i < detData.size(); i++)
        {
            KalmanTracker trk = KalmanTracker(detData[i].box, detData[i].title, detData[i].confidence);
            trackers.push_back(trk);
        }
        // output the first frame detections
        return inJNIArray;
    }


    ///////////////////////////////////////
    // 3.1. get predicted locations from existing trackers.

    for (auto it = trackers.begin(); it != trackers.end();)
    {
        Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else
        {
            it = trackers.erase(it);
        }
    }


    ///////////////////////////////////////
    // 3.2. associate detections to tracked object (both represented as bounding boxes)
    trkNum = predictedBoxes.size();
    detNum = detData.size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iouMatrix, assignment);


    // find matches, unmatched_detections and unmatched_predictions
    if (detNum > trkNum) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }

    // filter out matched with low IOU
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i])); // matchedPairs (tracking, detection)
    }


    ///////////////////////////////////////
    // 3.3. updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detData[detIdx].box);
    }


    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker = KalmanTracker(detData[umd].box, detData[umd].title, detData[umd].confidence);
        trackers.push_back(tracker);
    }


    // get trackers' output
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.confidence = (*it).m_confidence;
            res.title = (*it).m_title;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age)
            it = trackers.erase(it);
    }


    int outCArrayLen = frameTrackingResult.size() * 7;
    jfloat outCArray[outCArrayLen];

    for(int i = 0; i < frameTrackingResult.size(); i++){
        outCArray[i * 7 + 0] = frameTrackingResult[i].confidence;
        outCArray[i * 7 + 1] = frameTrackingResult[i].id;
        outCArray[i * 7 + 2] = frameTrackingResult[i].box.x;
        outCArray[i * 7 + 3] = frameTrackingResult[i].box.y;
        outCArray[i * 7 + 4] = frameTrackingResult[i].box.width;
        outCArray[i * 7 + 5] = frameTrackingResult[i].box.height;
        outCArray[i * 7 + 6] = frameTrackingResult[i].title;
    }


    // Step 3: Convert the C's Native jdouble[] to JNI jdoublearray, and return
    jfloatArray outJNIArray = env->NewFloatArray(outCArrayLen);  // allocate
    if (NULL == outJNIArray) return NULL;
    env->SetFloatArrayRegion(outJNIArray, 0 , outCArrayLen, outCArray);  // copy
    env->ReleaseFloatArrayElements(inJNIArray, inCArray, 0); // release resources
    return outJNIArray;
}
