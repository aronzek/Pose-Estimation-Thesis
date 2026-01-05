using UnityEngine;
using System.IO;

[RequireComponent(typeof(Camera))]
public class OrthoScreenshot : MonoBehaviour
{
    public Camera orthoCam;
    public int resolution = 2048;
    public string outputFile = "orthophoto.png";

    void Start()
    {
        TakeScreenshot();
    }

    public void TakeScreenshot()
    {
        RenderTexture rt = new RenderTexture(resolution, resolution, 24);
        orthoCam.targetTexture = rt;

        Texture2D screenshot = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
        orthoCam.Render();
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, resolution, resolution), 0, 0);
        orthoCam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        byte[] bytes = screenshot.EncodeToPNG();
        string path = Path.Combine(Application.dataPath, outputFile);
        File.WriteAllBytes(path, bytes);
        Debug.Log("Saved orthophoto to: " + path);
    }
}
