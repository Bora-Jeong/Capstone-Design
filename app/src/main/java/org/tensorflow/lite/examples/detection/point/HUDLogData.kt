package org.tensorflow.lite.examples.detection.point

data class HUDLogData(val timestamp:String, val hudPos:Point3d, val hudRot:Point3d, val posenetPos:Point3d, val leftScore:Float, val rightScore:Float, val oldposenetPos:Point3d, val obstaclePos:Point3d, val clusterCount:Int, val leftHandAnchor:Array<Float>, val rightHandAnchor:Array<Float>, val lasttimestamp:String){
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as HUDLogData

        if (timestamp != other.timestamp) return false
        if (hudPos != other.hudPos) return false
        if (hudRot != other.hudRot) return false
        if (posenetPos != other.posenetPos) return false
        if (obstaclePos != other.obstaclePos) return false
        if (clusterCount != other.clusterCount) return false
        if (!leftHandAnchor.contentEquals(other.leftHandAnchor)) return false
        if (!rightHandAnchor.contentEquals(other.rightHandAnchor)) return false
        if (lasttimestamp != other.lasttimestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = timestamp.hashCode()
        result = 31 * result + hudPos.hashCode()
        result = 31 * result + hudRot.hashCode()
        result = 31 * result + posenetPos.hashCode()
        result = 31 * result + obstaclePos.hashCode()
        result = 31 * result + clusterCount
        result = 31 * result + leftHandAnchor.contentHashCode()
        result = 31 * result + rightHandAnchor.contentHashCode()
        result = 31 * result + lasttimestamp.hashCode()
        return result
    }
}