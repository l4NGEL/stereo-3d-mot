#pragma once

namespace s3m {

/// The 80 COCO class names, index 0..79. `size` is set to 80.
inline const char* const* cocoClassNames(int& size) {
    static const char* const kNames[] = {
        "person",        "bicycle",       "car",           "motorcycle",    "airplane",
        "bus",           "train",         "truck",         "boat",          "traffic light",
        "fire hydrant",  "stop sign",     "parking meter", "bench",         "bird",
        "cat",           "dog",           "horse",         "sheep",         "cow",
        "elephant",      "bear",          "zebra",         "giraffe",       "backpack",
        "umbrella",      "handbag",       "tie",           "suitcase",      "frisbee",
        "skis",          "snowboard",     "sports ball",   "kite",          "baseball bat",
        "baseball glove","skateboard",    "surfboard",     "tennis racket", "bottle",
        "wine glass",    "cup",           "fork",          "knife",         "spoon",
        "bowl",          "banana",        "apple",         "sandwich",      "orange",
        "broccoli",      "carrot",        "hot dog",       "pizza",         "donut",
        "cake",          "chair",         "couch",         "potted plant",  "bed",
        "dining table",  "toilet",        "tv",            "laptop",        "mouse",
        "remote",        "keyboard",      "cell phone",    "microwave",     "oven",
        "toaster",       "sink",          "refrigerator",  "book",          "clock",
        "vase",          "scissors",      "teddy bear",    "hair drier",    "toothbrush"};
    size = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));
    return kNames;
}

/// COCO class name for `id`, or "?" when out of the 0..79 range. Never null.
inline const char* cocoClassName(int id) {
    int n = 0;
    const char* const* names = cocoClassNames(n);
    return (id >= 0 && id < n) ? names[id] : "?";
}

}  // namespace s3m
