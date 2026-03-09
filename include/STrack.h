#pragma once

#include <atomic>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(std::vector<float> tlwh_, float score, int cls = 0);
	~STrack();

	static std::vector<float> tlbr_to_tlwh(const std::vector<float> &tlbr);
	static void multi_predict(std::vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	std::vector<float> tlwh_to_xyah(const std::vector<float> &tlwh_tmp) const;
	std::vector<float> to_xyah() const;
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame() const;
	static void reset_id();

	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);
	cv::Scalar get_color() const;

public:
	bool is_activated;
	int track_id;
	int state;

	std::vector<float> _tlwh;
	std::vector<float> tlwh;
	std::vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;
	int cls;

private:
	byte_kalman::KalmanFilter kalman_filter;
	static std::atomic<int> _count;
};
