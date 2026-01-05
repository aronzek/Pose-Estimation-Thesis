using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;

public class DroneCapture : MonoBehaviour
{
    [Header("Main Settings")]
    public Camera droneCamera;
    public Transform geoRoot; // Should be at (0,0,0)
    public float overlapPercent = 70f;
    public float altitude = 60f;
    public string outputFolder = "DroneCaptures";
    public Vector2 groundSize = new Vector2(300f, 100f); // X (easting), Z (northing)

    [Header("Camera Settings")]
    public float fieldOfView = 60f;
    public int resolutionWidth = 1024;
    public int resolutionHeight = 1024;

    private int gridX, gridZ;
    private float imageFootprint;
    private List<CaptureMetadata> metadata = new List<CaptureMetadata>();

    void Start()
    {
        if (droneCamera == null) droneCamera = Camera.main;

        droneCamera.orthographic = false;
        droneCamera.fieldOfView = fieldOfView;
        droneCamera.orthographicSize = altitude * Mathf.Tan(fieldOfView * 0.5f * Mathf.Deg2Rad);
        droneCamera.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

        CalculateCoverage();
        StartCoroutine(CaptureRoutine());
    }

    void CalculateCoverage()
    {
        imageFootprint = 2f * altitude * Mathf.Tan(fieldOfView * 0.5f * Mathf.Deg2Rad);
        float spacing = imageFootprint * (1f - overlapPercent / 100f);
        gridX = Mathf.CeilToInt(groundSize.x / spacing);
        gridZ = Mathf.CeilToInt(groundSize.y / spacing);

        Debug.Log($"📸 Grid: {gridX}x{gridZ} (spacing ~{spacing:F1}m, footprint ~{imageFootprint:F1}m)");
    }

    IEnumerator CaptureRoutine()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string sessionPath = Path.Combine(outputFolder, $"Capture_{timestamp}");
        Directory.CreateDirectory(sessionPath);

        float spacing = imageFootprint * (1f - overlapPercent / 100f);

        for (int x = 0; x < gridX; x++)
        {
            for (int z = 0; z < gridZ; z++)
            {
                Vector3 localOffset = new Vector3(
                    x * spacing + imageFootprint / 2f,
                    altitude,
                    z * spacing + imageFootprint / 2f
                );

                Vector3 worldPos = geoRoot.position + localOffset;

                transform.position = worldPos;
                droneCamera.transform.localPosition = Vector3.zero;
                droneCamera.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);

                yield return null;

                RenderTexture rt = new RenderTexture(resolutionWidth, resolutionHeight, 24);
                droneCamera.targetTexture = rt;
                RenderTexture.active = rt;

                droneCamera.Render();

                Texture2D img = new Texture2D(resolutionWidth, resolutionHeight, TextureFormat.RGB24, false);
                img.ReadPixels(new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0);
                img.Apply();

                string filename = $"img_{x:00}_{z:00}.jpg";
                string filepath = Path.Combine(sessionPath, filename);
                File.WriteAllBytes(filepath, img.EncodeToJPG(95));

                float utmEasting = 409209.4f + localOffset.x;
                float utmNorthing = 5657397f + localOffset.z;

                metadata.Add(new CaptureMetadata
                {
                    filename = filename,
                    easting = utmEasting,
                    northing = utmNorthing,
                    altitude = altitude,
                    offset_from_geoRoot = new Vector3(localOffset.x, 0f, localOffset.z),
                    bounds = new ImageBounds
                    {
                        minX = utmEasting - imageFootprint / 2f,
                        maxX = utmEasting + imageFootprint / 2f,
                        minY = utmNorthing - imageFootprint / 2f,
                        maxY = utmNorthing + imageFootprint / 2f
                    }
                });

                droneCamera.targetTexture = null;
                RenderTexture.active = null;
                Destroy(rt);
                Destroy(img);
            }
        }

        SaveMetadata(sessionPath);
        Debug.Log("✅ Drone capture session complete.");
    }

    void SaveMetadata(string folder)
    {
        string json = JsonUtility.ToJson(new MetadataWrapper { captures = metadata }, true);
        File.WriteAllText(Path.Combine(folder, "capture_metadata.json"), json);
    }

    [System.Serializable]
    public class MetadataWrapper
    {
        public List<CaptureMetadata> captures;
    }

    [System.Serializable]
    public class CaptureMetadata
    {
        public string filename;
        public float easting;
        public float northing;
        public float altitude;
        public Vector3 offset_from_geoRoot;
        public ImageBounds bounds;
    }

    [System.Serializable]
    public class ImageBounds
    {
        public float minX;
        public float maxX;
        public float minY;
        public float maxY;
    }
}
