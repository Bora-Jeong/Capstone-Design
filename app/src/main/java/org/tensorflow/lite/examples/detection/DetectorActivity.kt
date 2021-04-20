/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.detection

import android.app.Activity
import android.graphics.*
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.ImageReader.OnImageAvailableListener
import android.os.Build
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.util.TypedValue
import android.view.Surface
import android.view.View
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import org.tensorflow.lite.examples.detection.customview.OverlayView
import org.tensorflow.lite.examples.detection.env.BorderedText
import org.tensorflow.lite.examples.detection.env.ImageUtils
import org.tensorflow.lite.examples.detection.env.Logger
import org.tensorflow.lite.examples.detection.pose.InferenceInfoGraphic
import org.tensorflow.lite.examples.detection.pose.PoseDetectorProcessor
import org.tensorflow.lite.examples.detection.tflite.Detector
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker
import java.io.IOException
import java.util.*

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(), OnImageAvailableListener {
    var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int? = null
    private var detector: Detector? = null
    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null
    private var computingDetection = false
    private var timestamp: Long = 0
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var tracker: MultiBoxTracker? = null
    private var borderedText: BorderedText? = null

    private var fpsTimer : Long = 0
    private val TEXT_COLOR = Color.WHITE
    private val TEXT_SIZE = 60.0f
    private val textPaint : Paint = Paint()
    init {
        textPaint.color = TEXT_COLOR
        textPaint.textSize = TEXT_SIZE
    }

    // pose estimation
    private val ORIENTATIONS = SparseIntArray()
    init {
        ORIENTATIONS.append(Surface.ROTATION_0, 0)
        ORIENTATIONS.append(Surface.ROTATION_90, 90)
        ORIENTATIONS.append(Surface.ROTATION_180, 180)
        ORIENTATIONS.append(Surface.ROTATION_270, 270)
    }

    private var poseDetectorProcessor : PoseDetectorProcessor? = null
    private var graphicOverlay: GraphicOverlay? = null


    public override fun onPreviewSizeChosen(size: Size?, rotation: Int) {
        val textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        var cropSize = TF_OD_API_INPUT_SIZE
        try {
            detector = TFLiteObjectDetectionAPIModel.create(
                    this,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED)
            cropSize = TF_OD_API_INPUT_SIZE
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing Detector!")
            val toast = Toast.makeText(
                    applicationContext, "Detector could not be initialized", Toast.LENGTH_SHORT)
            toast.show()
            finish()
        }
        previewWidth = size!!.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation)
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation!!, MAINTAIN_ASPECT)
        cropToFrameTransform = Matrix()
        frameToCropTransform?.invert(cropToFrameTransform)
        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback { canvas ->
            tracker!!.draw(canvas)
            if (isDebug) {
                tracker!!.drawDebug(canvas)
            }
        }
        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation!!)


        val options = PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
        poseDetectorProcessor = PoseDetectorProcessor(this, options, true, true, true, false, true)
        graphicOverlay = findViewById(R.id.pose_overlay)
        graphicOverlay!!.setImageSourceInfo(cropSize, cropSize, false)
    }

    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        //LOGGER.i("Preparing image $currTimestamp for detection in bg thread.")
        rgbFrameBitmap!!.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()

        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null)

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap)
        }
        runInBackground {
            LOGGER.i("Running detection on image $currTimestamp")
            val startTime = SystemClock.uptimeMillis()
            val results = detector!!.recognizeImage(croppedBitmap)
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap)
            val canvas = Canvas(cropCopyBitmap)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            if(graphicOverlay != null){
                poseDetectorProcessor?.processBitmap(croppedBitmap, graphicOverlay!!)
            }

            var minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API
            minimumConfidence = when (MODE) {
                DetectorMode.TF_OD_API -> MINIMUM_CONFIDENCE_TF_OD_API
                DetectorMode.YOLO4 -> MINIMUM_CONFIDENCE_YOLOv4
            }

            val mappedRecognitions: MutableList<Detector.Recognition> = ArrayList()
            for(result in results){
                if(result.location != null && result.confidence >= minimumConfidence){
                    mappedRecognitions.add(result)
                }
            }

            val trackedRecognitions = trackObjects(mappedRecognitions, timestamp)
            for(result in trackedRecognitions){
                val location = result.location
                canvas.drawRect(location, paint)
                cropToFrameTransform!!.mapRect(location)
                result.location = location
            }

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime

            tracker!!.trackResults(trackedRecognitions, currTimestamp)
            trackingOverlay!!.postInvalidate()
            computingDetection = false
            runOnUiThread {
                showFrameInfo(previewWidth.toString() + "x" + previewHeight)
                showCropInfo(cropCopyBitmap?.getWidth().toString() + "x" + cropCopyBitmap?.getHeight())
                if(fpsTimer > 0){
                    val sec = (SystemClock.uptimeMillis() - fpsTimer) * 0.001 // to sec
                    showInference("${(1 / sec).toInt()}")
                }
                fpsTimer = SystemClock.uptimeMillis()
            }
        }
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Throws(CameraAccessException::class)
    private fun getRotationCompensation(cameraId: String, activity: Activity, isFrontFacing: Boolean): Int {
        // Get the device's current rotation relative to its "native" orientation.
        // Then, from the ORIENTATIONS table, look up the angle the image must be
        // rotated to compensate for the device's rotation.
        val deviceRotation = activity.windowManager.defaultDisplay.rotation
        var rotationCompensation = ORIENTATIONS.get(deviceRotation)

        // Get the device's sensor orientation.
        val cameraManager = activity.getSystemService(CAMERA_SERVICE) as CameraManager
        val sensorOrientation = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.SENSOR_ORIENTATION)!!

        if (isFrontFacing) {
            rotationCompensation = (sensorOrientation + rotationCompensation) % 360
        } else { // back-facing
            rotationCompensation = (sensorOrientation - rotationCompensation + 360) % 360
        }
        return rotationCompensation
    }


    var titleArray = mutableListOf<String>()

    fun trackObjects(results: MutableList<Detector.Recognition>, frame: Long) : MutableList<Detector.Recognition> {

        if(results.size == 0) return  results

        var inputArray = FloatArray(results.size * 7);
        for(i in 0 until results.size){
            inputArray[i * 7 + 0] = results[i].confidence
            inputArray[i * 7 + 1] = results[i].id.toFloat()
            inputArray[i * 7 + 2] = results[i].location.left
            inputArray[i * 7 + 3] = results[i].location.top
            inputArray[i * 7 + 4] = results[i].location.width()
            inputArray[i * 7 + 5] = results[i].location.height()
            inputArray[i * 7 + 6] = titleArray.size.toFloat()
            titleArray.add(results[i].title)
        }


        var outputArray = sort(inputArray)

        var outputList: MutableList<Detector.Recognition> = ArrayList()
        for(i in 0 until outputArray.size / 7){
            var confidence = outputArray[i * 7 + 0]
            val id = outputArray[i * 7 + 1].toInt().toString()
            val x = outputArray[i * 7 + 2]
            val y = outputArray[i * 7 + 3]
            val width = outputArray[i * 7 + 4]
            val height = outputArray[i * 7 + 5]
            val title = outputArray[i * 7 + 6].toInt()
            var rcg = Detector.Recognition(id, titleArray[title], confidence, RectF(x, y, x + width, y + height));
            outputList.add(rcg)
        }

        return outputList
    }



    override val layoutId: Int
        protected get() = R.layout.tfe_od_camera_connection_fragment_tracking


    override val desiredPreviewFrameSize: Size?
        protected  get() = DESIRED_PREVIEW_SIZE

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API, YOLO4
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        runInBackground {
            try {
                detector!!.setUseNNAPI(isChecked)
            } catch (e: UnsupportedOperationException) {
                LOGGER.e(e, "Failed to set \"Use NNAPI\".")
                runOnUiThread { Toast.makeText(this, e.message, Toast.LENGTH_LONG).show() }
            }
        }
    }

    override fun setNumThreads(numThreads: Int) {
        runInBackground { detector!!.setNumThreads(numThreads) }
    }

    external fun sort(JNIArray: FloatArray): FloatArray


    companion object {
        init {
            System.loadLibrary("native-lib")
        }

        private val LOGGER = Logger()

        // Configuration values for the prepackaged SSD model.
        private const val TF_OD_API_INPUT_SIZE = 300
        private const val TF_OD_API_IS_QUANTIZED = true
        private const val TF_OD_API_MODEL_FILE = "detect.tflite"
        private const val TF_OD_API_LABELS_FILE = "labelmap.txt"

        private const val YOLOv4_INPUT_SIZE = 416
        private const val YOLOv4_IS_QUANTIZED = false
        private const val YOLOv4_MODEL_FILE = "yolov4.tflite"
        private const val YOLOv4_LABELS_FILE = "labelmap.txt"

        private val MODE = DetectorMode.TF_OD_API

        // Minimum detection confidence to track a detection.
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private const val MINIMUM_CONFIDENCE_YOLOv4 = 0.5f
        private const val MAINTAIN_ASPECT = false
        private val DESIRED_PREVIEW_SIZE = Size(640, 480)
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}