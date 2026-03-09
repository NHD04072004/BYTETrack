#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "BYTETracker.h"

namespace py = pybind11;

// Helper: convert tracked stracks to numpy array (N, 8) = [x1, y1, x2, y2, track_id, score, cls, state]
static py::array_t<float> stracks_to_numpy(const std::vector<STrack> &stracks) {
    py::array_t<float> result({(int)stracks.size(), 8});
    auto buf = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < (py::ssize_t)stracks.size(); i++) {
        buf(i, 0) = stracks[i].tlbr[0];
        buf(i, 1) = stracks[i].tlbr[1];
        buf(i, 2) = stracks[i].tlbr[2];
        buf(i, 3) = stracks[i].tlbr[3];
        buf(i, 4) = static_cast<float>(stracks[i].track_id);
        buf(i, 5) = stracks[i].score;
        buf(i, 6) = static_cast<float>(stracks[i].cls);
        buf(i, 7) = static_cast<float>(stracks[i].state);
    }
    return result;
}

// Helper: parse numpy detections to Object vector
static std::vector<Object> numpy_to_objects(py::array_t<float> dets, const std::string &format) {
    auto buf = dets.unchecked<2>();
    if (buf.shape(1) < 5) {
        throw std::runtime_error("Expected array with at least 5 columns: [x1, y1, x2/w, y2/h, score] or [x1, y1, x2/w, y2/h, score, label]");
    }
    std::vector<Object> objects;
    objects.reserve(buf.shape(0));
    bool has_label = buf.shape(1) >= 6;
    bool is_xyxy = (format == "xyxy");

    for (py::ssize_t i = 0; i < buf.shape(0); i++) {
        Object obj;
        if (is_xyxy) {
            // xyxy: x1, y1, x2, y2 -> x, y, w, h
            float x1 = buf(i, 0), y1 = buf(i, 1), x2 = buf(i, 2), y2 = buf(i, 3);
            obj.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
        } else {
            // xywh: x, y, w, h
            obj.rect = cv::Rect_<float>(buf(i, 0), buf(i, 1), buf(i, 2), buf(i, 3));
        }
        obj.prob = buf(i, 4);
        obj.label = has_label ? static_cast<int>(buf(i, 5)) : 0;
        objects.push_back(obj);
    }
    return objects;
}

PYBIND11_MODULE(_bytetrack, m) {
    m.doc() = "ByteTrack C++ multi-object tracker with pybind11 bindings";

    py::class_<Object>(m, "Object")
        .def(py::init<>())
        .def_readwrite("rect", &Object::rect)
        .def_readwrite("label", &Object::label)
        .def_readwrite("prob", &Object::prob);

    py::class_<cv::Rect_<float>>(m, "RectF")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &cv::Rect_<float>::x)
        .def_readwrite("y", &cv::Rect_<float>::y)
        .def_readwrite("width", &cv::Rect_<float>::width)
        .def_readwrite("height", &cv::Rect_<float>::height);

    py::enum_<TrackState>(m, "TrackState")
        .value("New", TrackState::New)
        .value("Tracked", TrackState::Tracked)
        .value("Lost", TrackState::Lost)
        .value("Removed", TrackState::Removed)
        .export_values();

    py::class_<STrack>(m, "STrack")
        .def(py::init<std::vector<float>, float, int>(),
            py::arg("tlwh"), py::arg("score"), py::arg("cls") = 0)
        .def_readonly("is_activated", &STrack::is_activated)
        .def_readonly("track_id", &STrack::track_id)
        .def_readonly("state", &STrack::state)
        .def_readonly("score", &STrack::score)
        .def_readonly("cls", &STrack::cls)
        .def_readonly("tlwh", &STrack::tlwh)
        .def_readonly("tlbr", &STrack::tlbr)
        .def_readonly("frame_id", &STrack::frame_id)
        .def_readonly("start_frame", &STrack::start_frame)
        .def_readonly("tracklet_len", &STrack::tracklet_len)
        .def("get_color", &STrack::get_color)
        .def("__repr__", [](const STrack &t) {
            return "<STrack id=" + std::to_string(t.track_id) +
                   " score=" + std::to_string(t.score) +
                   " cls=" + std::to_string(t.cls) +
                   " state=" + std::to_string(t.state) + ">";
        });

    py::class_<BYTETracker>(m, "BYTETracker")
        .def(py::init<int, float, float, float, float, bool, bool, float>(),
            py::arg("max_time_lost") = 15,
            py::arg("track_high_thresh") = 0.5f,
            py::arg("track_low_thresh") = 0.1f,
            py::arg("new_track_thresh") = 0.6f,
            py::arg("match_thresh") = 0.8f,
            py::arg("fuse_score") = true,
            py::arg("class_aware") = true,
            py::arg("min_box_area") = 0.0f)

        // Python properties instead of just setter methods
        .def_property("max_time_lost",
            &BYTETracker::get_max_time_lost, &BYTETracker::set_max_time_lost)
        .def_property("track_high_thresh",
            &BYTETracker::get_track_high_thresh, &BYTETracker::set_track_high_thresh)
        .def_property("track_low_thresh",
            &BYTETracker::get_track_low_thresh, &BYTETracker::set_track_low_thresh)
        .def_property("new_track_thresh",
            &BYTETracker::get_new_track_thresh, &BYTETracker::set_new_track_thresh)
        .def_property("match_thresh",
            &BYTETracker::get_match_thresh, &BYTETracker::set_match_thresh)
        .def_property("fuse_score",
            &BYTETracker::get_fuse_score, &BYTETracker::set_fuse_score)
        .def_property("class_aware",
            &BYTETracker::get_class_aware, &BYTETracker::set_class_aware)
        .def_property("min_box_area",
            &BYTETracker::get_min_box_area, &BYTETracker::set_min_box_area)
        .def_property_readonly("frame_id", &BYTETracker::get_frame_id)

        // Original update with Object list
        .def("update", [](BYTETracker &self, const std::vector<Object> &objects) {
            std::vector<STrack> output_stracks;
            std::vector<STrack> lost_stracks;
            self.update(objects, lost_stracks, output_stracks);
            return py::make_tuple(output_stracks, lost_stracks);
        }, py::arg("objects"),
           "Update tracker with detections. Returns (output_stracks, lost_stracks)")

        // Numpy input, STrack list output
        .def("update_from_numpy", [](BYTETracker &self, py::array_t<float> dets, const std::string &format) {
            auto objects = numpy_to_objects(dets, format);
            std::vector<STrack> output_stracks;
            std::vector<STrack> lost_stracks;
            self.update(objects, lost_stracks, output_stracks);
            return py::make_tuple(output_stracks, lost_stracks);
        }, py::arg("detections"), py::arg("format") = "xywh",
           "Update tracker with numpy array of shape (N, 5+).\n"
           "format='xywh': columns are [x, y, w, h, score, label(optional)]\n"
           "format='xyxy': columns are [x1, y1, x2, y2, score, label(optional)]\n"
           "Returns (output_stracks, lost_stracks)")

        // Numpy input, numpy output — for pure-numpy pipelines
        .def("update_numpy", [](BYTETracker &self, py::array_t<float> dets, const std::string &format) {
            auto objects = numpy_to_objects(dets, format);
            std::vector<STrack> output_stracks;
            std::vector<STrack> lost_stracks;
            self.update(objects, lost_stracks, output_stracks);
            return py::make_tuple(stracks_to_numpy(output_stracks), stracks_to_numpy(lost_stracks));
        }, py::arg("detections"), py::arg("format") = "xywh",
           "Update tracker with numpy array, returns numpy arrays.\n"
           "Input: (N, 5+) with format 'xywh' or 'xyxy'.\n"
           "Output: tuple of (tracked, lost) arrays of shape (M, 8)\n"
           "  columns: [x1, y1, x2, y2, track_id, score, cls, state]")

        .def("reset", &BYTETracker::reset)
        .def_static("reset_id", &BYTETracker::reset_id);

    // Helper to create Object from Python values
    m.def("make_object", [](float x, float y, float w, float h, float prob, int label) {
        Object obj;
        obj.rect = cv::Rect_<float>(x, y, w, h);
        obj.prob = prob;
        obj.label = label;
        return obj;
    }, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
       py::arg("prob"), py::arg("label") = 0,
       "Create an Object detection with (x, y, w, h, prob, label)");
}
