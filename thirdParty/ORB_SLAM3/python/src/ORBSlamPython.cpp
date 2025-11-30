#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <opencv2/core/core.hpp>
#include <pyboostcvconverter.hpp>
#include <KeyFrame.h>
#include <Converter.h>
#include <Tracking.h>
#include <MapPoint.h>
#include "ORBSlamPython.h"

static void* init_ar() {
    Py_Initialize();

    import_array();
    return NULL;
}

BOOST_PYTHON_MODULE(orbslam3)
{
    init_ar();

    boost::python::to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    boost::python::enum_<ORB_SLAM3::Tracking::eTrackingState>("TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::eTrackingState::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::eTrackingState::OK_KLT);
    
    boost::python::enum_<ORB_SLAM3::System::eSensor>("Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::eSensor::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::eSensor::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::eSensor::IMU_RGBD);

    boost::python::class_<ORBSlamPython, boost::noncopyable>("System", boost::python::init<const char*, const char*, boost::python::optional<ORB_SLAM3::System::eSensor, bool, bool, int, const char*>>())
        .def(boost::python::init<std::string, std::string, boost::python::optional<ORB_SLAM3::System::eSensor, bool, bool, int, std::string>>())
        .def("initialize", &ORBSlamPython::initialize)
        .def("load_and_process_mono", &ORBSlamPython::loadAndProcessMono)
        .def("process_image_mono", &ORBSlamPython::processMono)
        .def("load_and_process_stereo", &ORBSlamPython::loadAndProcessStereo)
        .def("process_image_stereo", &ORBSlamPython::processStereo)
        .def("load_and_process_rgbd", &ORBSlamPython::loadAndProcessRGBD)
        .def("process_image_rgbd", &ORBSlamPython::processRGBD)
        .def("shutdown", &ORBSlamPython::shutdown)
        .def("is_running", &ORBSlamPython::isRunning)
        .def("is_last_frame_kf", &ORBSlamPython::isLastFrameKeyframe)
        .def("reset", &ORBSlamPython::reset)
        .def("set_mode", &ORBSlamPython::setMode)
        .def("set_use_viewer", &ORBSlamPython::setUseViewer)
        .def("get_last_big_change_idx", &ORBSlamPython::getLastBigChangeIdx)
        .def("get_keyframe_points", &ORBSlamPython::getKeyframePoints)
        .def("get_last_trajectory_point", &ORBSlamPython::getLastTrajectoryPoint)
        .def("get_trajectory_points", &ORBSlamPython::getTrajectoryPoints)
        .def("get_tracked_mappoints", &ORBSlamPython::getTrackedMappoints)
        .def("get_tracking_state", &ORBSlamPython::getTrackingState)
        .def("get_num_features", &ORBSlamPython::getNumFeatures)
        .def("get_num_matched_features", &ORBSlamPython::getNumMatches)
        .def("save_settings", &ORBSlamPython::saveSettings)
        .def("load_settings", &ORBSlamPython::loadSettings)
        .def("save_settings_file", &ORBSlamPython::saveSettingsFile)
        .staticmethod("save_settings_file")
        .def("load_settings_file", &ORBSlamPython::loadSettingsFile)
        .staticmethod("load_settings_file");
}

ORBSlamPython::ORBSlamPython(std::string vocabFile, std::string settingsFile, ORB_SLAM3::System::eSensor sensorMode, const bool useViewer, const bool turnOffLC, const int initFr, std::string sequence)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(useViewer),
    bturnOffLC(turnOffLC),
    bUseRGB(true),
    initFr(initFr),
    sequence(sequence)
{
    
}

ORBSlamPython::ORBSlamPython(const char* vocabFile, const char* settingsFile, ORB_SLAM3::System::eSensor sensorMode, const bool useViewer, const bool turnOffLC, const int initFr, const char* sequence)
    : vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(useViewer),
    bturnOffLC(turnOffLC),
    bUseRGB(true),
    initFr(initFr),
    sequence(sequence)
{

}

ORBSlamPython::~ORBSlamPython()
{
    if (system)
    {
        system->Shutdown();
    }
}

bool ORBSlamPython::initialize()
{
    system = std::make_shared<ORB_SLAM3::System>(vocabluaryFile, settingsFile, sensorMode, bUseViewer, initFr, sequence, bturnOffLC);
    return true;
}

bool ORBSlamPython::isRunning()
{
    return system != nullptr;
}

void ORBSlamPython::reset()
{
    if (system)
    {
        system->Reset();
    }
}

bool ORBSlamPython::loadAndProcessMono(std::string imageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    return this->processMono(im, timestamp);
}

bool ORBSlamPython::processMono(cv::Mat image, double timestamp)
{
    if (!system || !image.data)
    {
        return false;
    }

    Sophus::SE3f pose = system->TrackMonocular(image, timestamp);
    //return !pose.empty();
    return system->GetTrackingState() == ORB_SLAM3::Tracking::eTrackingState::OK;
}

bool ORBSlamPython::loadAndProcessStereo(std::string leftImageFile, std::string rightImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat leftImage = cv::imread(leftImageFile, cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImageFile, cv::IMREAD_COLOR);
    if (bUseRGB) {
        cv::cvtColor(leftImage, leftImage, cv::COLOR_BGR2RGB);
        cv::cvtColor(rightImage, rightImage, cv::COLOR_BGR2RGB);
    }
    return this->processStereo(leftImage, rightImage, timestamp);
}

bool ORBSlamPython::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp)
{
    if (!system | !leftImage.data | !rightImage.data)
    {
        return false;
    }
    Sophus::SE3f pose = system->TrackStereo(leftImage, rightImage, timestamp);
    return system->GetTrackingState() == ORB_SLAM3::Tracking::eTrackingState::OK;
}

bool ORBSlamPython::loadAndProcessRGBD(std::string imageFile, std::string depthImageFile, double timestamp)
{
    if (!system)
    {
        return false;
    }
    cv::Mat im = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (bUseRGB)
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    }
    cv::Mat imDepth = cv::imread(depthImageFile, cv::IMREAD_UNCHANGED);
    return this->processRGBD(im, imDepth, timestamp);
}

bool ORBSlamPython::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp)
{
    if (!system | !image.data | !depthImage.data)
    {
        return false;
    }
    Sophus::SE3f pose = system->TrackRGBD(image, depthImage, timestamp);
    
    return system->GetTrackingState() == ORB_SLAM3::Tracking::eTrackingState::OK;
}

void ORBSlamPython::shutdown()
{
    if (system)
    {
        system->Shutdown();
        system.reset();
    }
}

ORB_SLAM3::Tracking::eTrackingState ORBSlamPython::getTrackingState() const
{
    if (system)
    {
        return static_cast<ORB_SLAM3::Tracking::eTrackingState>(system->GetTrackingState());
    }
    return ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY;
}

boost::python::list ORBSlamPython::getTrackedMappoints() const
{
    if (!system)
    {
        return boost::python::list();
    }
    
    vector<ORB_SLAM3::MapPoint*> Mps = system->GetTrackedMapPoints();// This call only get map points of current frame.
    
    boost::python::list map_points;
    for(size_t i=0; i<Mps.size(); i++)    {
        if (Mps[i] != NULL)
        {
        Eigen::Vector3f wp = Mps[i]->GetWorldPos();
        map_points.append(boost::python::make_tuple(
            wp(0),
            wp(1),
            wp(2)                          
            ));
        }
    }

    return map_points;
}

void ORBSlamPython::setMode(ORB_SLAM3::System::eSensor mode)
{
    sensorMode = mode;
}

void ORBSlamPython::setUseViewer(bool useViewer)
{
    bUseViewer = useViewer;
}

void ORBSlamPython::setRGBMode(bool rgb)
{
    bUseRGB = rgb;
}

bool ORBSlamPython::saveSettings(boost::python::dict settings) const
{
    return ORBSlamPython::saveSettingsFile(settings, settingsFile);
}

boost::python::dict ORBSlamPython::loadSettings() const
{
    return ORBSlamPython::loadSettingsFile(settingsFile);
}

boost::python::dict ORBSlamPython::loadSettingsFile(std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::READ);
    cv::FileNode root = fs.root();
    if (root.isMap()) 
    {
        return readMap(root);
    }
    else if (root.isSeq())
    {
        boost::python::dict settings;
        settings["root"] = readSequence(root);
        return settings;
    }
    return boost::python::dict();
}

bool ORBSlamPython::saveSettingsFile(boost::python::dict settings, std::string settingsFilename)
{
    cv::FileStorage fs(settingsFilename.c_str(), cv::FileStorage::WRITE);
    
    boost::python::list keys = settings.keys();
    for (int index = 0; index < boost::python::len(keys); ++index)
    {
        boost::python::extract<std::string> extractedKey(keys[index]);
        if (!extractedKey.check())
        {
            continue;
        }
        std::string key = extractedKey;
        
        boost::python::extract<int> intValue(settings[key]);
        if (intValue.check())
        {
            fs << key << int(intValue);
            continue;
        }
        
        boost::python::extract<float> floatValue(settings[key]);
        if (floatValue.check())
        {
            fs << key << float(floatValue);
            continue;
        }
        
        boost::python::extract<std::string> stringValue(settings[key]);
        if (stringValue.check())
        {
            fs << key << std::string(stringValue);
            continue;
        }
    }
    
    return true;
}

////////////////////
unsigned int ORBSlamPython::getLastBigChangeIdx() const
{
    if (!system)
    {
        return 0;
    }
    return system->GetLastBigChangeIdx();
}


boost::python::list ORBSlamPython::getKeyframePoints() const
{
    if (!system)
    {
        return boost::python::list();
    }

    vector<ORB_SLAM3::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.

    boost::python::list trajectory;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

        if(pKF->isBad())
            continue;

        Eigen::Matrix3f R = pKF->GetRotation().transpose(); //.t();
        Eigen::Vector3f t = pKF->GetCameraCenter();
        trajectory.append(boost::python::make_tuple(
                              pKF->mTimeStamp,
                                R(0,0), R(0,1), R(0,2), t(0),
                                R(1,0), R(1,1), R(1,2), t(1),
                                R(2,0), R(2,1), R(2,2), t(2)
                              ));
    }

    return trajectory;
}

bool ORBSlamPython::isLastFrameKeyframe()
{   
    // This assumes that is never run in parallel with Tracking
    if (!system)
    {
        return false;
    }
    return system->GetTracker()->isLastFrameKeyframe();
}

unsigned int ORBSlamPython::getNumFeatures() const
{
    if (system)
    {
        return system->GetTracker()->mCurrentFrame.mvKeys.size();
    }
    return 0;
}

unsigned int ORBSlamPython::getNumMatches() const
{
    if (system)
    {
        // This code is based on the display code in FrameDrawer.cc, with a little extra safety logic to check the length of the vectors.
        ORB_SLAM3::Tracking* pTracker = system->GetTracker();
        unsigned int matches = 0;
        unsigned int num = pTracker->mCurrentFrame.mvKeys.size();
        if (pTracker->mCurrentFrame.mvpMapPoints.size() < num)
        {
            num = pTracker->mCurrentFrame.mvpMapPoints.size();
        }
        if (pTracker->mCurrentFrame.mvbOutlier.size() < num)
        {
            num = pTracker->mCurrentFrame.mvbOutlier.size();
        }
        for(unsigned int i = 0; i < num; ++i)
        {
            ORB_SLAM3::MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP && !pTracker->mCurrentFrame.mvbOutlier[i] && pMP->Observations() > 0)
            {
                ++matches;
            }
        }
        return matches;
    }
    return 0;
}

boost::python::tuple ORBSlamPython::getLastTrajectoryPoint() const
{
    if (!system)
    {
        return boost::python::tuple();
    }

    ORB_SLAM3::Tracking* tracker = system->GetTracker();

    // Check if the trajectory lists are empty. If so, return an empty tuple.
    if (tracker->mlRelativeFramePoses.empty() ||
        tracker->mlpReferences.empty() ||
        tracker->mlFrameTimes.empty()) {
        return boost::python::tuple();
    }

    // Safely get the last elements from the tracker's lists.
    // Note: This still doesn't solve the race condition if called concurrently 
    // with the tracking thread modifying these lists. A proper fix for the race
    // condition would involve mutex protection or a system-level thread-safe getter.
    const Sophus::SE3f& lastRelPose = tracker->mlRelativeFramePoses.back();
    ORB_SLAM3::KeyFrame* pKF_ref = tracker->mlpReferences.back();
    double lastTimestamp = tracker->mlFrameTimes.back();

    vector<ORB_SLAM3::KeyFrame*> vpKFs = system->GetKeyFrames(); // This is thread-safe
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

    Sophus::SE3f Two;
    if (!vpKFs.empty()) {
        Two = vpKFs[0]->GetPoseInverse();
    }

    ORB_SLAM3::KeyFrame* pKF = pKF_ref;

    Sophus::SE3f Trw;

    while(pKF != NULL && pKF->isBad())
    {
        ORB_SLAM3::KeyFrame* pKFParent;

        Trw = Trw * pKF->mTcp;
        pKFParent = pKF->GetParent();
        if (pKFParent == pKF) {
            // We've found a frame that is it's own parent, presumably a root or something. Break out
            break;
        } else {
            pKF = pKFParent;
        }
    }
    if (pKF == NULL || pKF->isBad()) {
        return boost::python::tuple();
    }
    
    Trw = Trw*pKF->GetPose()*Two;

    Sophus::SE3f Tcw = lastRelPose*Trw;
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Matrix3f Rwc = Twc.rotationMatrix();
    Eigen::Vector3f twc = Twc.translation();
    return boost::python::make_tuple(
        lastTimestamp,
                                Rwc(0,0), Rwc(0,1), Rwc(0,2), twc(0),
                                Rwc(1,0), Rwc(1,1), Rwc(1,2), twc(1),
                                Rwc(2,0), Rwc(2,1), Rwc(2,2), twc(2)
    );
}

boost::python::list ORBSlamPython::getTrajectoryPoints() const
{
    if (!system)
    {
        return boost::python::list();
    }
    
    vector<ORB_SLAM3::KeyFrame*> vpKFs = system->GetKeyFrames();
    std::sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // Of course, if we have no keyframes, then just use the identity matrix.
    Sophus::SE3f Two;
    if (vpKFs.size() > 0) {
        Two = vpKFs[0]->GetPoseInverse();
    }

    boost::python::list trajectory;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<ORB_SLAM3::KeyFrame*>::iterator lRit = system->GetTracker()->mlpReferences.begin();
    std::list<double>::iterator lT = system->GetTracker()->mlFrameTimes.begin();
    //for(std::list<cv::Mat>::iterator lit=system->GetTracker()->mlRelativeFramePoses.begin(), lend=system->GetTracker()->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    std::list<bool>::iterator lbL = system->GetTracker()->mlbLost.begin();
    for(std::list<Sophus::SE3f>::iterator lit=system->GetTracker()->mlRelativeFramePoses.begin(), lend=system->GetTracker()->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;
        while(pKF != NULL && pKF->isBad())
        {
            ORB_SLAM3::KeyFrame* pKFParent;

            Trw = Trw * pKF->mTcp;
            pKFParent = pKF->GetParent();
            if (pKFParent == pKF) {
                // We've found a frame that is it's own parent, presumably a root or something. Break out
                break;
            } else {
                pKF = pKFParent;
            }
        }
        if (pKF != NULL && !pKF->isBad()) {
            Trw = Trw * pKF->GetPose() * Two;

            Sophus::SE3f Tcw = (*lit)*Trw;
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Matrix3f Rwc = Twc.rotationMatrix();
            Eigen::Vector3f twc = Twc.translation();

            trajectory.append(boost::python::make_tuple(
                                *lT,
                                Rwc(0,0), Rwc(0,1), Rwc(0,2), twc(0),
                                Rwc(1,0), Rwc(1,1), Rwc(1,2), twc(1),
                                Rwc(2,0), Rwc(2,1), Rwc(2,2), twc(2)
                            ));
        }
    }

    return trajectory;
}


// ----------- HELPER DEFINITIONS -----------
boost::python::dict readMap(cv::FileNode fn, int depth)
{
    boost::python::dict map;
    if (fn.isMap()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            std::string key = item.name();
            
            if (item.isNone())
            {
                map[key] = boost::python::object();
            }
            else if (item.isInt())
            {
                map[key] = int(item);
            }
            else if (item.isString())
            {
                map[key] = std::string(item);
            }
            else if (item.isReal())
            {
                map[key] = float(item);
            }
            else if (item.isSeq() && depth > 0)
            {
                map[key] = readSequence(item, depth-1);
            }
            else if (item.isMap() && depth > 0)
            {
                map[key] = readMap(item, depth-1);  // Depth-limited recursive call to read inner maps
            }
        }
    }
    return map;
}

boost::python::list readSequence(cv::FileNode fn, int depth)
{
    boost::python::list sequence;
    if (fn.isSeq()) {
        cv::FileNodeIterator it = fn.begin(), itEnd = fn.end();
        for (; it != itEnd; ++it) {
            cv::FileNode item = *it;
            
            if (item.isNone())
            {
                sequence.append(boost::python::object());
            }
            else if (item.isInt())
            {
                sequence.append(int(item));
            }
            else if (item.isString())
            {
                sequence.append(std::string(item));
            }
            else if (item.isReal())
            {
                sequence.append(float(item));
            }
            else if (item.isSeq() && depth > 0)
            {
                sequence.append(readSequence(item, depth-1)); // Depth-limited recursive call to read nested sequences
            }
            else if (item.isMap() && depth > 0)
            {
                sequence.append(readMap(item, depth-1));
            }
        }
    }
    return sequence;
}
