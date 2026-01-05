using UnityEngine;

public class GeoRootManager : MonoBehaviour
{
    // The real-world UTM coordinates (meters) of the SW corner of your scene
    public float utmEasting = 409209.42f;
    public float utmNorthing = 5657397.23f;
    public float elevationOffset = 0f; // Set this to average water level, like 156.95f if needed

    public static Vector3 geoOffset;

    void Awake()
    {
        // This sets the static geoOffset once when the scene starts
        geoOffset = new Vector3(utmEasting, elevationOffset, utmNorthing);
    }

    public static Vector3 UTMToUnity(Vector3 utmPos)
    {
        return new Vector3(
            utmPos.x - geoOffset.x,
            utmPos.y - geoOffset.y,
            utmPos.z - geoOffset.z
        );
    }

    public static Vector3 UnityToUTM(Vector3 unityPos)
    {
        return new Vector3(
            unityPos.x + geoOffset.x,
            unityPos.y + geoOffset.y,
            unityPos.z + geoOffset.z
        );
    }
}