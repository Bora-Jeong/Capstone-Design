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

import android.Manifest
import android.app.Activity
import android.app.Fragment
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.database.Cursor
import android.hardware.Camera
import android.hardware.Camera.PreviewCallback
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.net.Uri
import android.opengl.Matrix
import android.os.*
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.Surface
import android.view.View
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.view.WindowManager
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.appcompat.widget.Toolbar
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import org.tensorflow.lite.examples.detection.env.ImageUtils
import org.tensorflow.lite.examples.detection.env.Logger
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import com.github.nkzawa.emitter.Emitter
import com.github.nkzawa.engineio.client.transports.WebSocket
import com.github.nkzawa.socketio.client.IO
import com.github.nkzawa.socketio.client.IO.socket
import com.github.nkzawa.socketio.client.Socket
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.io.File
import kotlin.math.atan
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.tensorflow.lite.examples.detection.point.*
import org.tensorflow.lite.examples.detection.pose.FrameMetadata
import org.tensorflow.lite.examples.detection.pose.PoseDetectorProcessor
import java.net.URISyntaxException
import java.nio.ByteBuffer

abstract class CameraActivity : AppCompatActivity(), OnImageAvailableListener, PreviewCallback, CompoundButton.OnCheckedChangeListener, View.OnClickListener {
    protected var previewWidth = 0
    protected var previewHeight = 0
    val isDebug = false
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var useCamera2API = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    protected var luminanceStride = 0
        private set
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private var bottomSheetLayout: LinearLayout? = null
    private var gestureLayout: LinearLayout? = null
    private var sheetBehavior: BottomSheetBehavior<LinearLayout?>? = null
    protected var frameValueTextView: TextView? = null
    protected var cropValueTextView: TextView? = null
    protected var inferenceTimeTextView: TextView? = null
    protected var bottomSheetArrowImageView: ImageView? = null
    private var plusImageView: ImageView? = null
    private var minusImageView: ImageView? = null
    private var apiSwitchCompat: SwitchCompat? = null
    private var threadsTextView: TextView? = null

    // posent code
    private var isPointCloudDataLoaded = false
    var pointCloudData: PointCloudData? = null
    val jacksonMapper = jacksonObjectMapper()
    var fovX = 0.0f
    var fovY = 0.0f
    var imageDimensions = IntArray(2)
    val MODELVIEWPROJECTION_MAT_ROW = 4
    val MODELVIEWPROJECTION_MAT_COLUMN = 4
    val modelViewProjection = FloatArray(16)
    var matModelViewProjection = Mat(
            MODELVIEWPROJECTION_MAT_ROW,
            MODELVIEWPROJECTION_MAT_COLUMN,
            CvType.CV_32FC1
    )
    val invModelViewProjection = Mat(
            MODELVIEWPROJECTION_MAT_ROW,
            MODELVIEWPROJECTION_MAT_COLUMN,
            CvType.CV_32FC1
    )
    var matInvRT = Mat(4, 4, CvType.CV_32FC1)
    var matRT = Mat(4, 4, CvType.CV_32FC1)
    private var cameraToWorld = Mat(4, 4, CvType.CV_32FC1)
    private var worldToCamera = Mat(4, 4, CvType.CV_32FC1)
    val ROTATION_MAT_ROW = 3
    val ROTATION_MAT_COLUMN = 3
    var matRotation = Mat(ROTATION_MAT_ROW, ROTATION_MAT_COLUMN, CvType.CV_32FC1)

    val TRANSLATION_MAT_ROW = 3
    val TRANSLATION_MAT_COLUMN = 1
    val matTranslation = Mat(TRANSLATION_MAT_ROW, TRANSLATION_MAT_COLUMN, CvType.CV_32FC1)

    val INTRINSIC_CAM_MAT_ROW = 3
    val INTRINSIC_CAM_MAT_COLUMN = 3
    val matCamIntrinsic = Mat(INTRINSIC_CAM_MAT_ROW, INTRINSIC_CAM_MAT_COLUMN, CvType.CV_32FC1)

    val INVERSE_RM_MAT_ROW = 3
    val INVERSE_RM_MAT_COLUMN = 3
    val invRM = Mat(INVERSE_RM_MAT_ROW, INVERSE_RM_MAT_COLUMN, CvType.CV_32FC1)

    val INVERSE_RT_MAT_ROW = 3
    val INVERSE_RT_MAT_COLUMN = 1
    val invR_x_tvec = Mat(INVERSE_RT_MAT_ROW, INVERSE_RT_MAT_COLUMN, CvType.CV_32FC1)

    val camPos = Point3d()
    var surfWidth = 0
    var surfHeight = 0
    private val planeMgr = PlaneMgr()
    val baseDepthPadding = 0.1f
    private val pointCloud = mutableListOf<Point3d>()
    private val clusterPointCloud = mutableListOf<MutableList<Point3d>>()
    val logPath = Environment.getExternalStorageDirectory().toString() + "/PoseNetLog"
    val logDir = File(logPath)
    val logDataPoseNet = mutableListOf<List<String>>()
    val logDataHUD = mutableListOf<List<String>>()
    var baseTimestamp:Long = 0
    private var mSocket: Socket? = null




    override fun onCreate(savedInstanceState: Bundle?) {
        LOGGER.d("onCreate $this")
        super.onCreate(null)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.tfe_od_activity_camera)
        //val toolbar = findViewById<Toolbar>(R.id.toolbar)
        //setSupportActionBar(toolbar)
        //supportActionBar!!.setDisplayShowTitleEnabled(false)
        if (hasPermission()) {
            setFragment()
        } else {
            requestPermission()
        }
        threadsTextView = findViewById(R.id.threads)
        plusImageView = findViewById(R.id.plus)
        minusImageView = findViewById(R.id.minus)
        apiSwitchCompat = findViewById(R.id.api_info_switch)
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout)
        gestureLayout = findViewById(R.id.gesture_layout)
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout)
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow)
        val vto = gestureLayout?.getViewTreeObserver()
        if (vto != null) {
            vto.addOnGlobalLayoutListener(
                    object : OnGlobalLayoutListener {
                        override fun onGlobalLayout() {
                            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                                gestureLayout?.getViewTreeObserver()?.removeGlobalOnLayoutListener(this)
                            } else {
                                gestureLayout?.getViewTreeObserver()?.removeOnGlobalLayoutListener(this)
                            }
                            //                int width = bottomSheetLayout.getMeasuredWidth();
                            val height = gestureLayout!!.getMeasuredHeight()
                            sheetBehavior?.setPeekHeight(height)
                        }
                    })
        }
        sheetBehavior?.setHideable(false)
        sheetBehavior?.setBottomSheetCallback(
                object : BottomSheetCallback() {
                    override fun onStateChanged(bottomSheet: View, newState: Int) {
                        when (newState) {
                            BottomSheetBehavior.STATE_HIDDEN -> {
                            }
                            BottomSheetBehavior.STATE_EXPANDED -> {
                                bottomSheetArrowImageView?.setImageResource(R.drawable.icn_chevron_down)
                            }
                            BottomSheetBehavior.STATE_COLLAPSED -> {
                                bottomSheetArrowImageView?.setImageResource(R.drawable.icn_chevron_up)
                            }
                            BottomSheetBehavior.STATE_DRAGGING -> {
                            }
                            BottomSheetBehavior.STATE_SETTLING -> bottomSheetArrowImageView?.setImageResource(R.drawable.icn_chevron_up)
                        }
                    }

                    override fun onSlide(bottomSheet: View, slideOffset: Float) {}
                })
        frameValueTextView = findViewById(R.id.frame_info)
        cropValueTextView = findViewById(R.id.crop_info)
        inferenceTimeTextView = findViewById(R.id.inference_info)
        apiSwitchCompat?.setOnCheckedChangeListener(this)
        plusImageView?.setOnClickListener(this)
        minusImageView?.setOnClickListener(this)

        //findViewById<Button>(R.id.btnImport).text = stringFromJNI()


        findViewById<View>(R.id.btnImport).setOnClickListener { view: View? ->
            if(!isPointCloudDataLoaded) {
                openPointCloud()
                try {
                    val opts = IO.Options()
                    opts.port = 3000
                    opts.transports = arrayOf(WebSocket.NAME)
                    mSocket = socket("http://172.16.201.243:3000", opts)
                }catch (e: URISyntaxException){
                    Log.d("SocketIO", e.message)
                }
                mSocket!!.on(Socket.EVENT_CONNECT, onConnect)
                mSocket!!.on(Socket.EVENT_CONNECT_ERROR, onConnectError);
                mSocket!!.on(Socket.EVENT_CONNECT_TIMEOUT, onConnectError);
                try {
                    mSocket!!.connect()
                    mSocket!!.on("receive hud data", onHUDMsg)
                }catch (e: Exception){
                    Log.d("SocketIO", e.message)
                }
            }else{
                exportLog()
            }
        }
    }



    override fun onActivityResult(requestCode: Int, resultCode: Int, resultData: Intent?) {
        super.onActivityResult(requestCode, resultCode, resultData)
        if (requestCode == READ_REQUEST_CODE && resultCode == RESULT_OK && resultData != null) {
            val uri = resultData.data ?: return
            this.applicationContext.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            val jsonFile = File(getPathFromUri(uri))
            try {
                pointCloudData = jacksonMapper.readValue<PointCloudData>(jsonFile)
                handlePointCloudData()

                findViewById<Button>(R.id.btnImport)?.text = getString(R.string.ExportLog)
                val logLinePoseNet = mutableListOf<String>()

                logLinePoseNet.add("Timestamp")
                logLinePoseNet.add("LeftPos x")
                logLinePoseNet.add("LeftPos y")
                logLinePoseNet.add("LeftPos z")
                logLinePoseNet.add("RightPos x")
                logLinePoseNet.add("RightPos y")
                logLinePoseNet.add("RightPos z")
                logLinePoseNet.add("CenterPos x")
                logLinePoseNet.add("CenterPos y")
                logLinePoseNet.add("CenterPos z")
                logLinePoseNet.add("Old CenterPos x")
                logLinePoseNet.add("Old CenterPos y")
                logLinePoseNet.add("Old CenterPos z")
                logLinePoseNet.add("left score")
                logLinePoseNet.add("right score")
                logLinePoseNet.add("NearPoint x")
                logLinePoseNet.add("NearPoint y")
                logLinePoseNet.add("NearPoint z")
                logLinePoseNet.add("Left2DPos x")
                logLinePoseNet.add("Left2DPos y")
                logLinePoseNet.add("Right2DPos x")
                logLinePoseNet.add("Right2DPos y")
                logDataPoseNet.add(logLinePoseNet.toList())

                val logLineHUD = mutableListOf<String>()
                logLineHUD.add("Log Timestamp")
                logLineHUD.add("Timestamp")
                logLineHUD.add("HUDPos x")
                logLineHUD.add("HUDPos y")
                logLineHUD.add("HUDPos z")
                logLineHUD.add("HUDRot x")
                logLineHUD.add("HUDRot y")
                logLineHUD.add("HUDRot z")
                logLineHUD.add("HUDRot w")
                logLineHUD.add("PoseNetPos x")
                logLineHUD.add("PoseNetPos y")
                logLineHUD.add("PoseNetPos z")
                logLineHUD.add("Old PoseNetPos x")
                logLineHUD.add("Old PoseNetPos y")
                logLineHUD.add("Old PoseNetPos z")
                logLineHUD.add("left score")
                logLineHUD.add("right score")
                logLineHUD.add("Obstacle x")
                logLineHUD.add("Obstacle y")
                logLineHUD.add("Obstacle z")
                logLineHUD.add("cluster count")
                logLineHUD.add("leftHandAnchor x")
                logLineHUD.add("leftHandAnchor y")
                logLineHUD.add("leftHandAnchor z")
                logLineHUD.add("rightHandAnchor x")
                logLineHUD.add("rightHandAnchor y")
                logLineHUD.add("rightHandAnchor z")
                logLineHUD.add("last timestamp")

                logDataHUD.add(logLineHUD.toList())

                baseTimestamp = System.currentTimeMillis()

            } catch (e: Exception) {
                Toast.makeText(this.applicationContext, "PointCloud load fail", Toast.LENGTH_LONG).show()
                Log.d("PoseNet", e.message.toString())
            }
        }
    }

    private fun getPathFromUri(uri: Uri): String? {
        val context = this.applicationContext
        if(DocumentsContract.isDocumentUri(context, uri)){
            if (isExternalStorageDocument(uri)){
                val docId = DocumentsContract.getDocumentId(uri)
                val split = docId.split(":").toTypedArray()
                val type = split[0]

                if ("primary".equals(type, ignoreCase = true)) {
                    return Environment.getExternalStorageDirectory()
                            .toString() + "/" + split[1]
                }
            }else if (isDownloadsDocument(uri)) {
                val id = DocumentsContract.getDocumentId(uri)
                val contentUri = ContentUris.withAppendedId(
                        Uri.parse("content://downloads/public_downloads"), id.toLong()
                )

                return getDataColumn(context!!, contentUri, null, null)!!
            }else if (isMediaDocument(uri)) {
                val docId = DocumentsContract.getDocumentId(uri)
                val split = docId.split(":")
                val type = split[0]

                var contentUri:Uri? = null
                when (type) {
                    "image" -> {
                        contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                    }
                    "video" -> {
                        contentUri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                    }
                    "audio" -> {
                        contentUri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
                    }
                }

                val selection = "_id=?"
                val selectionArgs:Array<String?>? = arrayOf(split[1])

                return getDataColumn(context!!, contentUri, selection, selectionArgs)!!
            }
        }else if ("content".equals(uri.scheme, ignoreCase = true)) {
            return getDataColumn(context!!, uri, null, null)!!
        }
        // File
        else if ("file".equals(uri.scheme, ignoreCase = true)) {
            return uri.path!!
        }
        return ""
    }


    // json 파일을 import 한 후 실행시키는 함수
    private fun handlePointCloudData(){
        if (pointCloudData != null){
            calcFov(pointCloudData!!.cameraIntrinsicsData.focalLength[0])
            mkMatCamIntrinsic(
                    pointCloudData!!.cameraIntrinsicsData.focalLength,
                    pointCloudData!!.cameraIntrinsicsData.principalPoint
            )
            imageDimensions = pointCloudData!!.cameraIntrinsicsData.imageDimensions.toIntArray()
            val viewMatrix = pointCloudData!!.cameraMatrix.toFloatArray()
            for (i in 0 until MODELVIEWPROJECTION_MAT_ROW){
                for (j in 0 until MODELVIEWPROJECTION_MAT_COLUMN){
                    matRT.put(
                            i,
                            j,
                            arrayOf(viewMatrix[i * MODELVIEWPROJECTION_MAT_COLUMN + j]).toFloatArray()
                    )
                }
            }
            cameraToWorld = matRT;
            worldToCamera = cameraToWorld.inv()
            matRT = matRT.t()

            mkMatRT()
            mkInverseMatrix()
            Matrix.multiplyMM(
                    modelViewProjection,
                    0,
                    pointCloudData!!.projectionMatrix.toFloatArray(),
                    0,
                    pointCloudData!!.cameraMatrix.toFloatArray(),
                    0
            )


            for (i in 0 until MODELVIEWPROJECTION_MAT_ROW){
                for (j in 0 until MODELVIEWPROJECTION_MAT_COLUMN){
                    matModelViewProjection.put(
                            i,
                            j,
                            arrayOf(modelViewProjection[i * MODELVIEWPROJECTION_MAT_COLUMN + j]).toFloatArray()
                    )
                }
            }
            matModelViewProjection = matModelViewProjection.t()
            Core.invert(matModelViewProjection, invModelViewProjection)

            for (plane in pointCloudData!!.planeData) {
                val points = mutableListOf<Pair<Point2d, Point3d>>()
                for (point in plane.polygon) {
                    val pointMat = FloatArray(3)
                    pointMat[0] = point.x
                    pointMat[1] = point.y
                    pointMat[2] = point.z
                    val igPoint = imageGeometry(pointMat)
                    points.add(
                            Pair(
                                    Point2d(igPoint[0], igPoint[1]), Point3d(
                                    pointMat[0],
                                    pointMat[1],
                                    pointMat[2]
                            )
                            )
                    )
                }
                planeMgr.add(depth = plane.centerPosition[1], points = points.toTypedArray())
            }
            planeMgr.sort()
            val baseDepth = planeMgr.baseDepth()
            val pointCloudArray = mutableListOf<Float>()
            for (clusterPoints in pointCloudData!!.clusterData) {
                val subClusterPointCloud = mutableListOf<Point3d>()
                for (point in clusterPoints){
                    if (baseDepth + baseDepthPadding < point.y){
                        pointCloud.add(point)
                        subClusterPointCloud.add(point)
                        pointCloudArray.add(point.x)
                        pointCloudArray.add(point.y)
                        pointCloudArray.add(point.z)
                        pointCloudArray.add(point.confidence)
                    }
                }
                clusterPointCloud.add(subClusterPointCloud)
            }

            val clusterIndexes = mutableListOf<Int>()
            for (cluster in clusterPointCloud){
                clusterIndexes.add(cluster.size)
            }

//            CreateOctree(pointCloud = pointCloudArray.toFloatArray(),
//                    clusterIndex = clusterIndexes.toIntArray()
//            )

            isPointCloudDataLoaded = true

        }
    }

    private fun imageGeometry(pointMat: FloatArray):FloatArray{
        val igPoint = FloatArray(2)
        val point = Point3()
        point.x = pointMat[0].toDouble()
        point.y = pointMat[1].toDouble()
        point.z = pointMat[2].toDouble()
        val matPoint = MatOfPoint3f(point)

        val rvec = Mat(3, 1, CvType.CV_32FC1)
        Calib3d.Rodrigues(matRotation, rvec) // convert a rotation matrix to a rotation vector or vice versa

        val imagePoint = MatOfPoint2f()
        Calib3d.projectPoints(    // project 3D points to an image plane.
                matPoint,
                rvec,
                matTranslation,
                matCamIntrinsic,
                MatOfDouble(),
                imagePoint,
                Mat(),
                surfWidth.toDouble() / surfHeight.toDouble()
        )

        igPoint[0] = imagePoint.get(0, 0)[0].toFloat()
        igPoint[1] = imagePoint.get(0, 0)[1].toFloat()

        return igPoint
    }

    private fun mkInverseMatrix(){
        Core.gemm(matRotation.inv(), matCamIntrinsic.inv(), 1.0, Mat(), 0.0, invRM)
        Core.gemm(matRotation.inv(), matTranslation, 1.0, Mat(), 0.0, invR_x_tvec)
        matInvRT = matRT.inv()
    }

    private fun mkMatRT(){
        matRotation.put(0, 0, arrayListOf(matRT.get(0, 0)[0].toFloat()).toFloatArray())
        matRotation.put(0, 1, arrayListOf(matRT.get(0, 1)[0].toFloat()).toFloatArray())
        matRotation.put(0, 2, arrayListOf(matRT.get(0, 2)[0].toFloat()).toFloatArray())

        matRotation.put(1, 0, arrayListOf(matRT.get(1, 0)[0].toFloat()).toFloatArray())
        matRotation.put(1, 1, arrayListOf(matRT.get(1, 1)[0].toFloat()).toFloatArray())
        matRotation.put(1, 2, arrayListOf(matRT.get(1, 2)[0].toFloat()).toFloatArray())

        matRotation.put(2, 0, arrayListOf(matRT.get(2, 0)[0].toFloat()).toFloatArray())
        matRotation.put(2, 1, arrayListOf(matRT.get(2, 1)[0].toFloat()).toFloatArray())
        matRotation.put(2, 2, arrayListOf(matRT.get(2, 2)[0].toFloat()).toFloatArray())

        matTranslation.put(0, 0, arrayListOf(matRT.get(0, 3)[0].toFloat()).toFloatArray())
        matTranslation.put(1, 0, arrayListOf(matRT.get(1, 3)[0].toFloat()).toFloatArray())
        matTranslation.put(2, 0, arrayListOf(matRT.get(2, 3)[0].toFloat()).toFloatArray())

        camPos.x = matRT.get(0, 3)[0].toFloat()
        camPos.y = matRT.get(1, 3)[0].toFloat()
        camPos.z = matRT.get(2, 3)[0].toFloat()
    }


    private fun mkMatCamIntrinsic(focalLength: Array<Float>, principalPoint: Array<Float>){
        matCamIntrinsic.put(0, 0, arrayListOf(focalLength[0]).toFloatArray())
        matCamIntrinsic.put(0, 1, arrayListOf(0.0f).toFloatArray())
        matCamIntrinsic.put(0, 2, arrayListOf(principalPoint[0]).toFloatArray())

        matCamIntrinsic.put(1, 0, arrayListOf(0.0f).toFloatArray())
        matCamIntrinsic.put(1, 1, arrayListOf(focalLength[1]).toFloatArray())
        matCamIntrinsic.put(1, 2, arrayListOf(principalPoint[1]).toFloatArray())

        matCamIntrinsic.put(2, 0, arrayListOf(0.0f).toFloatArray())
        matCamIntrinsic.put(2, 1, arrayListOf(0.0f).toFloatArray())
        matCamIntrinsic.put(2, 2, arrayListOf(1.0f).toFloatArray())
    }

    private fun calcFov(focalLength: Float){
        fovX = Math.toDegrees((2 * atan(imageDimensions[0] / (focalLength * 2.0f)).toDouble())).toFloat()
        fovY = Math.toDegrees((2 * atan(imageDimensions[1] / (focalLength * 2.0f)).toDouble())).toFloat()
    }

    private val onConnect: Emitter.Listener = Emitter.Listener { args ->
        run {
            Log.d("SocketIO", "connected")
        }
    }

    private val onConnectError: Emitter.Listener = Emitter.Listener { args ->
        run {
            Log.d("SocketIO", args[0].toString())
        }
    }

    private fun exportLog(){
        if (logDir.exists() || logDir.mkdir()){
            //var postfix = LocalDateTime.now().toString()
            var postfix = "hi"
            postfix = postfix.replace(":", "")
            val logDataFilePoseNet = "$logPath/log_$postfix" + "_posenet.csv"
            val logDataFileHUD = "$logPath/log_$postfix" + "_hud.csv"
            csvWriter().writeAll(logDataPoseNet.toList(), logDataFilePoseNet)
            csvWriter().writeAll(logDataHUD.toList(), logDataFileHUD)
        }
    }

    private val onHUDMsg = Emitter.Listener { args ->
        if (args.isNotEmpty()) {
            val hudLogData = jacksonMapper.readValue<HUDLogData>(args[0].toString())
            val dataList = mutableListOf<String>()
            //dataList.add(LocalDateTime.now().toString())
            dataList.add(hudLogData.timestamp)
            dataList.add(hudLogData.hudPos.x.toString())
            dataList.add(hudLogData.hudPos.y.toString())
            dataList.add(hudLogData.hudPos.z.toString())
            dataList.add(hudLogData.hudRot.x.toString())
            dataList.add(hudLogData.hudRot.y.toString())
            dataList.add(hudLogData.hudRot.z.toString())
            dataList.add(hudLogData.hudRot.confidence.toString())
            dataList.add(hudLogData.posenetPos.x.toString())
            dataList.add(hudLogData.posenetPos.y.toString())
            dataList.add(hudLogData.posenetPos.z.toString())
            dataList.add(hudLogData.oldposenetPos.x.toString())
            dataList.add(hudLogData.oldposenetPos.y.toString())
            dataList.add(hudLogData.oldposenetPos.z.toString())
            dataList.add(hudLogData.leftScore.toString())
            dataList.add(hudLogData.rightScore.toString())
            dataList.add(hudLogData.obstaclePos.x.toString())
            dataList.add(hudLogData.obstaclePos.y.toString())
            dataList.add(hudLogData.obstaclePos.z.toString())
            dataList.add(hudLogData.clusterCount.toString())
            dataList.add(hudLogData.leftHandAnchor[0].toString())
            dataList.add(hudLogData.leftHandAnchor[1].toString())
            dataList.add(hudLogData.leftHandAnchor[2].toString())
            dataList.add(hudLogData.rightHandAnchor[0].toString())
            dataList.add(hudLogData.rightHandAnchor[1].toString())
            dataList.add(hudLogData.rightHandAnchor[2].toString())
            dataList.add(hudLogData.lasttimestamp)

            logDataHUD.add(dataList.toList())
        }
    }


    private fun isExternalStorageDocument(uri: Uri): Boolean {
        return "com.android.externalstorage.documents" == uri.authority
    }

    private fun isDownloadsDocument(uri: Uri): Boolean {
        return "com.android.providers.downloads.documents" == uri.authority
    }

    private fun isMediaDocument(uri: Uri): Boolean {
        return "com.android.providers.media.documents" == uri.authority
    }

    private fun getDataColumn(context: Context, uri: Uri?, selection: String?, selectionArgs: Array<String?>?): String? {
        var cursor: Cursor? = null
        val column = "_data"
        val projection = arrayOf(column)
        try {
            cursor = context.contentResolver.query(
                    uri!!, projection, selection, selectionArgs,
                    null
            )
            if (cursor != null && cursor.moveToFirst()) {
                val column_index = cursor.getColumnIndexOrThrow(column)
                return cursor.getString(column_index)
            }
        } finally {
            cursor!!.close()
        }
        return null
    }


    private fun openPointCloud() {
        val intent =
                Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }

        startActivityForResult(intent, READ_REQUEST_CODE)
    }


    protected fun getRgbBytes(): IntArray? {
        imageConverter!!.run()
        return rgbBytes
    }

    protected val luminance: ByteArray?
        protected get() = yuvBytes[0]

    /** Callback for android.hardware.Camera API  */
    override fun onPreviewFrame(bytes: ByteArray, camera: Camera) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!")
            return
        }
        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                val previewSize = camera.parameters.previewSize
                previewHeight = previewSize.height
                previewWidth = previewSize.width
                rgbBytes = IntArray(previewWidth * previewHeight)
                onPreviewSizeChosen(Size(previewSize.width, previewSize.height), 90)
            }
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            return
        }
        isProcessingFrame = true
        yuvBytes[0] = bytes
        luminanceStride = previewWidth
        imageConverter = Runnable { ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes) }

        postInferenceCallback = Runnable {
            camera.addCallbackBuffer(bytes)
            isProcessingFrame = false
        }
        processImage()
    }

    /** Callback for Camera2 API  */
    override fun onImageAvailable(reader: ImageReader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return

            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            Trace.beginSection("imageAvailable")
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = object : Runnable {
                override fun run() {
                    ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0],
                            yuvBytes[1],
                            yuvBytes[2],
                            previewWidth,
                            previewHeight,
                            luminanceStride,
                            uvRowStride,
                            uvPixelStride,
                            rgbBytes)
                }
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }

            processImage()

        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()
    }

    public override fun onStart() {
        super.onStart()
        LOGGER.d("onStart $this")
        val permissionStorage = ContextCompat.checkSelfPermission(
               applicationContext,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
        if (permissionStorage != PackageManager.PERMISSION_GRANTED) {
            //requestStoragePermission()
        }
        super.onStart()
    }


    public override fun onResume() {
        super.onResume()
        LOGGER.d("onResume $this")
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    public override fun onPause() {
        super.onPause()
        LOGGER.d("onPause $this")
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }
        super.onPause()
    }

    public override fun onStop() {
        super.onStop()
        LOGGER.d("onStop $this")
        super.onStop()
    }

    public override fun onDestroy() {
        super.onDestroy()
        LOGGER.d("onDestroy $this")
    }

    protected fun runInBackground(r: Runnable?) {
        if (handler != null) {
            handler!!.post(r)
        }
    }

    override fun onRequestPermissionsResult(
            requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment()
            } else {
                requestPermission()
            }
        }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                        this@CameraActivity,
                        "Camera permission is required for this demo",
                        Toast.LENGTH_LONG)
                        .show()
            }
            requestPermissions(arrayOf(PERMISSION_CAMERA), PERMISSIONS_REQUEST)
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
            characteristics: CameraCharacteristics, requiredLevel: Int): Boolean {
        val deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
    }

    private fun chooseCamera(): String? {
        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }
                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        ?: continue

                // Fallback to camera1 API for internal cameras that don't have full support.
                // This should help with legacy situations where using the camera2 API causes
                // distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL
                        || isHardwareLevelSupported(
                        characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL))
                LOGGER.i("Camera API lv2?: %s", useCamera2API)
                return cameraId
            }
        } catch (e: CameraAccessException) {
            LOGGER.e(e, "Not allowed to access camera")
        }
        return null
    }

    protected fun setFragment() {
        val cameraId = chooseCamera()
        val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment = CameraConnectionFragment.newInstance(
                    { size, rotation ->
                        previewHeight = size.height
                        previewWidth = size.width
                        onPreviewSizeChosen(size, rotation)
                    },
                    this,
                    layoutId,
                    desiredPreviewFrameSize)
            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment = LegacyCameraConnectionFragment(this, layoutId, desiredPreviewFrameSize)
        }
        fragmentManager.beginTransaction().replace(R.id.container, fragment).commit()
    }

    protected fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity())
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
    }

    protected fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }

    protected val screenOrientation: Int
        protected get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }

    override fun onCheckedChanged(buttonView: CompoundButton, isChecked: Boolean) {
        setUseNNAPI(isChecked)
        if (isChecked) apiSwitchCompat!!.text = "NNAPI" else apiSwitchCompat!!.text = "TFLITE"
    }

    override fun onClick(v: View) {
        if (v.id == R.id.plus) {
            val threads: String = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads: Int = threads.toInt()
            if (numThreads >= 9) return
            numThreads++
            threadsTextView!!.text = numThreads.toString()
            setNumThreads(numThreads)
        } else if (v.id == R.id.minus) {
            val threads: String = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads: Int = threads.toInt()
            if (numThreads == 1) {
                return
            }
            numThreads--
            threadsTextView!!.text = numThreads.toString()
            setNumThreads(numThreads)
        }
    }

    // private external fun CreateOctree(pointCloud: FloatArray, clusterIndex: IntArray)
    //private external fun SearchNearestPoint(point:FloatArray):FloatArray
   //  private external fun SearchNearestPoints(point:FloatArray):FloatArray


    protected fun showFrameInfo(frameInfo: String?) {
        frameValueTextView!!.text = frameInfo
    }

    protected fun showCropInfo(cropInfo: String?) {
        cropValueTextView!!.text = cropInfo
    }

    protected fun showInference(inferenceTime: String?) {
        inferenceTimeTextView!!.text = inferenceTime
    }


    protected abstract fun processImage()
    protected abstract fun onPreviewSizeChosen(size: Size?, rotation: Int)
    protected abstract val layoutId: Int
    protected abstract val desiredPreviewFrameSize: Size?
    protected abstract fun setNumThreads(numThreads: Int)
    protected abstract fun setUseNNAPI(isChecked: Boolean)


    external fun stringFromJNI(): String

    companion object {
        init {
            System.loadLibrary("native-lib")
            System.loadLibrary("KalmanTracker")
        }
        private val LOGGER = Logger()
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private fun allPermissionsGranted(grantResults: IntArray): Boolean {
            for (result in grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    return false
                }
            }
            return true
        }
    }
}