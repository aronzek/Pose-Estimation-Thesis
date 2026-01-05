using UnityEngine;

[System.Serializable]
public class ContainerInfo
{
    public float[] position;    // [x, y, z]
    public float rotationY;

    public ContainerInfo(Vector3 pos, Vector3 size, float rotY)
    {
        this.position = new float[] { pos.x, pos.y, pos.z };
        this.rotationY = rotY;
    }
}