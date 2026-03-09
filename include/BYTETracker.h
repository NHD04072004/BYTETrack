#pragma once

#include "STrack.h"
#include <unordered_map>
#include <unordered_set>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class BYTETracker
{
public:
	BYTETracker(int max_time_lost = 15, float track_high_thresh = 0.5,
				float track_low_thresh = 0.1, float new_track_thresh = 0.6,
				float match_thresh = 0.8, bool fuse_score = true,
				bool class_aware = true, float min_box_area = 0.0f);
	~BYTETracker();

	void update(const std::vector<Object>& objects,
				std::vector<STrack>& lost_stracks,
				std::vector<STrack>& output_stracks);

	// Setters
	void set_max_time_lost(int val) { max_time_lost = val; }
	void set_track_high_thresh(float thresh) { track_high_thresh = thresh; }
	void set_track_low_thresh(float thresh) { track_low_thresh = thresh; }
	void set_new_track_thresh(float thresh) { new_track_thresh = thresh; }
	void set_match_thresh(float thresh) { match_thresh = thresh; }
	void set_fuse_score(bool val) { fuse_score_ = val; }
	void set_class_aware(bool val) { class_aware_ = val; }
	void set_min_box_area(float val) { min_box_area_ = val; }

	// Getters
	int get_max_time_lost() const { return max_time_lost; }
	float get_track_high_thresh() const { return track_high_thresh; }
	float get_track_low_thresh() const { return track_low_thresh; }
	float get_new_track_thresh() const { return new_track_thresh; }
	float get_match_thresh() const { return match_thresh; }
	bool get_fuse_score() const { return fuse_score_; }
	bool get_class_aware() const { return class_aware_; }
	float get_min_box_area() const { return min_box_area_; }
	int get_frame_id() const { return frame_id; }

	void reset();
	static void reset_id();

private:
	std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

	std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

	void linear_assignment(std::vector<std::vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int>> &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float>> iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size, bool fuse_score = true);
	std::vector<std::vector<float>> iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
	std::vector<std::vector<float>> ious(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs);

	double lapjv(const std::vector<std::vector<float>> &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
	float track_high_thresh;
	float track_low_thresh;
	float new_track_thresh;
	float match_thresh;
	bool fuse_score_;
	bool class_aware_;
	float min_box_area_;

	int frame_id;
	int max_time_lost;

	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};
