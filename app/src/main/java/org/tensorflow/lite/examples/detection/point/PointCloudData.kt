package org.tensorflow.lite.examples.detection.point

data class PointCloudData(val cameraIntrinsicsData: CameraIntrinsicsData, val cameraPosition:Array<Float>, val displayOrientedPose:Array<Float>, val projectionMatrix:Array<Float>, val cameraMatrix:Array<Float>, val planeData:Array<PlaneData>, val clusterData:Array<Array<Point3d>>, val pointData:Array<Point3d>, val downsampledPointData:Array<Point3d>, val filteredPointData:Array<Point3d>) {
    data class CameraIntrinsicsData(val focalLength:Array<Float>, val imageDimensions:Array<Int>, val principalPoint:Array<Float>) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as CameraIntrinsicsData

            if (!focalLength.contentEquals(other.focalLength)) return false
            if (!imageDimensions.contentEquals(other.imageDimensions)) return false
            if (!principalPoint.contentEquals(other.principalPoint)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = focalLength.contentHashCode()
            result = 31 * result + imageDimensions.contentHashCode()
            result = 31 * result + principalPoint.contentHashCode()
            return result
        }
    }

    data class PlaneData(val centerPosition:Array<Float>, val distance:Float, val externX:Float, val externZ:Float, val polygon:Array<Point3d>){
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as PlaneData

            if (!centerPosition.contentEquals(other.centerPosition)) return false
            if (distance != other.distance) return false
            if (externX != other.externX) return false
            if (externZ != other.externZ) return false
            if (!polygon.contentEquals(other.polygon)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = centerPosition.contentHashCode()
            result = 31 * result + distance.hashCode()
            result = 31 * result + externX.hashCode()
            result = 31 * result + externZ.hashCode()
            result = 31 * result + polygon.contentHashCode()
            return result
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PointCloudData

        if (cameraIntrinsicsData != other.cameraIntrinsicsData) return false
        if (!cameraPosition.contentEquals(other.cameraPosition)) return false
        if (!displayOrientedPose.contentEquals(other.displayOrientedPose)) return false
        if (!projectionMatrix.contentEquals(other.projectionMatrix)) return false
        if (!cameraMatrix.contentEquals(other.cameraMatrix)) return false
        if (!planeData.contentEquals(other.planeData)) return false
        if (!clusterData.contentDeepEquals(other.clusterData)) return false
        if (!pointData.contentEquals(other.pointData)) return false
        if (!downsampledPointData.contentEquals(other.downsampledPointData)) return false
        if (!filteredPointData.contentEquals(other.filteredPointData)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = cameraIntrinsicsData.hashCode()
        result = 31 * result + cameraPosition.contentHashCode()
        result = 31 * result + displayOrientedPose.contentHashCode()
        result = 31 * result + projectionMatrix.contentHashCode()
        result = 31 * result + cameraMatrix.contentHashCode()
        result = 31 * result + planeData.contentHashCode()
        result = 31 * result + clusterData.contentDeepHashCode()
        result = 31 * result + pointData.contentHashCode()
        result = 31 * result + downsampledPointData.contentHashCode()
        result = 31 * result + filteredPointData.contentHashCode()
        return result
    }
}