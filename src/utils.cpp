#include "BYTETracker.h"
#include "lapjv.h"
#include <stdexcept>
#include <memory>

// FIX: Use unordered_set instead of map, fix logic bug (!exists[tid] was inserting key)
std::vector<STrack*> BYTETracker::joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb)
{
	std::unordered_set<int> exists;
	std::vector<STrack*> res;
	for (size_t i = 0; i < tlista.size(); i++)
	{
		exists.insert(tlista[i]->track_id);
		res.push_back(tlista[i]);
	}
	for (size_t i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (exists.find(tid) == exists.end())
		{
			exists.insert(tid);
			res.push_back(&tlistb[i]);
		}
	}
	return res;
}

std::vector<STrack> BYTETracker::joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb)
{
	std::unordered_set<int> exists;
	std::vector<STrack> res;
	for (size_t i = 0; i < tlista.size(); i++)
	{
		exists.insert(tlista[i].track_id);
		res.push_back(tlista[i]);
	}
	for (size_t i = 0; i < tlistb.size(); i++)
	{
		int tid = tlistb[i].track_id;
		if (exists.find(tid) == exists.end())
		{
			exists.insert(tid);
			res.push_back(tlistb[i]);
		}
	}
	return res;
}

// FIX: Use unordered_set instead of map<int, STrack> (avoids copying STrack into map)
std::vector<STrack> BYTETracker::sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb)
{
	std::unordered_set<int> remove_ids;
	for (size_t i = 0; i < tlistb.size(); i++)
	{
		remove_ids.insert(tlistb[i].track_id);
	}

	std::vector<STrack> res;
	for (size_t i = 0; i < tlista.size(); i++)
	{
		if (remove_ids.find(tlista[i].track_id) == remove_ids.end())
		{
			res.push_back(tlista[i]);
		}
	}
	return res;
}

// FIX: Use unordered_set instead of linear find() for O(1) lookup
void BYTETracker::remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb)
{
	std::vector<std::vector<float>> pdist = iou_distance(stracksa, stracksb);
	std::vector<std::pair<int, int>> pairs;
	for (size_t i = 0; i < pdist.size(); i++)
	{
		for (size_t j = 0; j < pdist[i].size(); j++)
		{
			if (pdist[i][j] < 0.15)
			{
				pairs.push_back(std::make_pair(i, j));
			}
		}
	}

	std::unordered_set<int> dupa, dupb;
	for (size_t i = 0; i < pairs.size(); i++)
	{
		int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
		int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
		if (timep > timeq)
			dupb.insert(pairs[i].second);
		else
			dupa.insert(pairs[i].first);
	}

	for (size_t i = 0; i < stracksa.size(); i++)
	{
		if (dupa.find(i) == dupa.end())
		{
			resa.push_back(stracksa[i]);
		}
	}

	for (size_t i = 0; i < stracksb.size(); i++)
	{
		if (dupb.find(i) == dupb.end())
		{
			resb.push_back(stracksb[i]);
		}
	}
}

void BYTETracker::linear_assignment(std::vector<std::vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
	std::vector<std::vector<int>> &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b)
{
	if (cost_matrix.empty())
	{
		for (int i = 0; i < cost_matrix_size; i++)
		{
			unmatched_a.push_back(i);
		}
		for (int i = 0; i < cost_matrix_size_size; i++)
		{
			unmatched_b.push_back(i);
		}
		return;
	}

	std::vector<int> rowsol;
	std::vector<int> colsol;
	lapjv(cost_matrix, rowsol, colsol, true, thresh);
	for (size_t i = 0; i < rowsol.size(); i++)
	{
		if (rowsol[i] >= 0)
		{
			std::vector<int> match = {(int)i, rowsol[i]};
			matches.push_back(match);
		}
		else
		{
			unmatched_a.push_back(i);
		}
	}

	for (size_t i = 0; i < colsol.size(); i++)
	{
		if (colsol[i] < 0)
		{
			unmatched_b.push_back(i);
		}
	}
}

std::vector<std::vector<float>> BYTETracker::ious(std::vector<std::vector<float>> &atlbrs, std::vector<std::vector<float>> &btlbrs)
{
	std::vector<std::vector<float>> ious;
	if (atlbrs.empty() || btlbrs.empty())
		return ious;

	ious.resize(atlbrs.size());
	for (size_t i = 0; i < ious.size(); i++)
	{
		ious[i].resize(btlbrs.size(), 0.0f);
	}

	for (size_t k = 0; k < btlbrs.size(); k++)
	{
		float box_area = (btlbrs[k][2] - btlbrs[k][0]) * (btlbrs[k][3] - btlbrs[k][1]);
		for (size_t n = 0; n < atlbrs.size(); n++)
		{
			float iw = std::min(atlbrs[n][2], btlbrs[k][2]) - std::max(atlbrs[n][0], btlbrs[k][0]);
			if (iw > 0)
			{
				float ih = std::min(atlbrs[n][3], btlbrs[k][3]) - std::max(atlbrs[n][1], btlbrs[k][1]);
				if (ih > 0)
				{
					float ua = (atlbrs[n][2] - atlbrs[n][0]) * (atlbrs[n][3] - atlbrs[n][1]) + box_area - iw * ih;
					ious[n][k] = iw * ih / ua;
				}
			}
		}
	}

	return ious;
}

std::vector<std::vector<float>> BYTETracker::iou_distance(
	std::vector<STrack*> &atracks,
	std::vector<STrack> &btracks,
	int &dist_size,
	int &dist_size_size,
	bool fuse_score)
{
	std::vector<std::vector<float>> cost_matrix;
	if (atracks.empty() || btracks.empty())
	{
		dist_size = atracks.size();
		dist_size_size = btracks.size();
		return cost_matrix;
	}
	std::vector<std::vector<float>> atlbrs, btlbrs;
	atlbrs.reserve(atracks.size());
	btlbrs.reserve(btracks.size());
	for (size_t i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i]->tlbr);
	}
	for (size_t i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	dist_size = atracks.size();
	dist_size_size = btracks.size();

	std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);

	for (size_t i = 0; i < _ious.size(); i++)
	{
		std::vector<float> _iou;
		_iou.reserve(_ious[i].size());
		for (size_t j = 0; j < _ious[i].size(); j++)
		{
			float sim = _ious[i][j];
			if (fuse_score)
			{
				sim *= btracks[j].score;
			}
			_iou.push_back(1.0f - sim);
		}
		cost_matrix.push_back(std::move(_iou));
	}

	return cost_matrix;
}

std::vector<std::vector<float>> BYTETracker::iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks)
{
	std::vector<std::vector<float>> atlbrs, btlbrs;
	atlbrs.reserve(atracks.size());
	btlbrs.reserve(btracks.size());
	for (size_t i = 0; i < atracks.size(); i++)
	{
		atlbrs.push_back(atracks[i].tlbr);
	}
	for (size_t i = 0; i < btracks.size(); i++)
	{
		btlbrs.push_back(btracks[i].tlbr);
	}

	std::vector<std::vector<float>> _ious = ious(atlbrs, btlbrs);
	std::vector<std::vector<float>> cost_matrix;
	for (size_t i = 0; i < _ious.size(); i++)
	{
		std::vector<float> _iou;
		_iou.reserve(_ious[i].size());
		for (size_t j = 0; j < _ious[i].size(); j++)
		{
			_iou.push_back(1.0f - _ious[i][j]);
		}
		cost_matrix.push_back(std::move(_iou));
	}

	return cost_matrix;
}

double BYTETracker::lapjv(const std::vector<std::vector<float>> &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
	bool extend_cost, float cost_limit, bool return_cost)
{
	std::vector<std::vector<float>> cost_c;
	cost_c.assign(cost.begin(), cost.end());

	std::vector<std::vector<float>> cost_c_extended;

	int n_rows = cost.size();
	int n_cols = cost[0].size();
	rowsol.resize(n_rows);
	colsol.resize(n_cols);

	int n = 0;
	if (n_rows == n_cols)
	{
		n = n_rows;
	}
	else
	{
		if (!extend_cost)
		{
			// FIX: throw exception instead of system("pause") + exit(0)
			throw std::runtime_error("lapjv: cost matrix is not square. Set extend_cost=true.");
		}
	}

	if (extend_cost || cost_limit < LONG_MAX)
	{
		n = n_rows + n_cols;
		cost_c_extended.resize(n);
		for (int i = 0; i < n; i++)
			cost_c_extended[i].resize(n);

		if (cost_limit < LONG_MAX)
		{
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					cost_c_extended[i][j] = cost_limit / 2.0;
				}
			}
		}
		else
		{
			float cost_max = -1;
			for (size_t i = 0; i < cost_c.size(); i++)
			{
				for (size_t j = 0; j < cost_c[i].size(); j++)
				{
					if (cost_c[i][j] > cost_max)
						cost_max = cost_c[i][j];
				}
			}
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					cost_c_extended[i][j] = cost_max + 1;
				}
			}
		}

		for (int i = n_rows; i < n; i++)
		{
			for (int j = n_cols; j < n; j++)
			{
				cost_c_extended[i][j] = 0;
			}
		}
		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				cost_c_extended[i][j] = cost_c[i][j];
			}
		}

		cost_c.clear();
		cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
	}

	// FIX: correct allocation sizes (was sizeof(type) * n, now just n)
	// Use unique_ptr for exception-safe memory management
	std::unique_ptr<double*[]> cost_ptr(new double*[n]);
	for (int i = 0; i < n; i++)
		cost_ptr[i] = new double[n];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cost_ptr[i][j] = cost_c[i][j];
		}
	}

	std::unique_ptr<int[]> x_c(new int[n]);
	std::unique_ptr<int[]> y_c(new int[n]);

	int ret = lapjv_internal(n, cost_ptr.get(), x_c.get(), y_c.get());
	if (ret != 0)
	{
		// FIX: clean up and throw instead of system("pause") + exit(0)
		for (int i = 0; i < n; i++)
			delete[] cost_ptr[i];
		throw std::runtime_error("lapjv: internal solver failed");
	}

	double opt = 0.0;

	if (n != n_rows)
	{
		for (int i = 0; i < n; i++)
		{
			if (x_c[i] >= n_cols)
				x_c[i] = -1;
			if (y_c[i] >= n_rows)
				y_c[i] = -1;
		}
		for (int i = 0; i < n_rows; i++)
		{
			rowsol[i] = x_c[i];
		}
		for (int i = 0; i < n_cols; i++)
		{
			colsol[i] = y_c[i];
		}

		if (return_cost)
		{
			for (size_t i = 0; i < rowsol.size(); i++)
			{
				if (rowsol[i] != -1)
				{
					opt += cost_ptr[i][rowsol[i]];
				}
			}
		}
	}
	else if (return_cost)
	{
		for (size_t i = 0; i < rowsol.size(); i++)
		{
			opt += cost_ptr[i][rowsol[i]];
		}
	}

	for (int i = 0; i < n; i++)
	{
		delete[] cost_ptr[i];
	}

	return opt;
}
