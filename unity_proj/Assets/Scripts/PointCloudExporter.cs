using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class TexturePointCloudExporter : MonoBehaviour
{
    public GameObject targetObject;
    public int samplesPerTriangle = 30;
    public string fileName = "textured_container.ply";
    public bool useVertexColors = false; // Toggle between vertex colors and texture sampling

    [ContextMenu("EXPORT WITH TEXTURE COLORS")]
    void ExportWithTextureColors()
    {
        if (!targetObject) return;

        MeshFilter mf = targetObject.GetComponent<MeshFilter>();
        MeshRenderer mr = targetObject.GetComponent<MeshRenderer>();
        if (!mf || !mr) return;

        List<Vector3> points = new List<Vector3>();
        List<Color> colors = new List<Color>();
        Mesh mesh = mf.sharedMesh;

        // Get texture if available
        Texture2D tex = mr.sharedMaterial.mainTexture as Texture2D;
        if (tex != null && !tex.isReadable)
        {
            Debug.LogWarning("Texture not readable! Enable Read/Write in import settings");
            return;
        }

        Vector3[] vertices = mesh.vertices;
        Vector2[] uvs = mesh.uv;
        int[] triangles = mesh.triangles;

        for (int i = 0; i < triangles.Length; i += 3)
        {
            Vector3 v1 = vertices[triangles[i]];
            Vector3 v2 = vertices[triangles[i + 1]];
            Vector3 v3 = vertices[triangles[i + 2]];
            Vector2 uv1 = uvs[triangles[i]];
            Vector2 uv2 = uvs[triangles[i + 1]];
            Vector2 uv3 = uvs[triangles[i + 2]];

            for (int j = 0; j < samplesPerTriangle; j++)
            {
                // Barycentric coordinates
                float r1 = Random.Range(0f, 1f);
                float r2 = Random.Range(0f, 1f);
                if (r1 + r2 > 1)
                {
                    r1 = 1 - r1;
                    r2 = 1 - r2;
                }

                // Calculate point position
                Vector3 point = v1 + r1 * (v2 - v1) + r2 * (v3 - v1);
                points.Add(point);

                // Calculate color
                if (useVertexColors && mesh.colors.Length > 0)
                {
                    colors.Add(mesh.colors[triangles[i]]);
                }
                else if (tex != null)
                {
                    Vector2 uv = uv1 + r1 * (uv2 - uv1) + r2 * (uv3 - uv1);
                    colors.Add(tex.GetPixelBilinear(uv.x, uv.y));
                }
                else
                {
                    colors.Add(mr.sharedMaterial.color);
                }
            }
        }

        SaveColoredPLY(points, colors);
    }

    void SaveColoredPLY(List<Vector3> points, List<Color> colors)
    {
        string path = Path.Combine(Application.dataPath, "..", fileName);
        using (StreamWriter writer = new StreamWriter(path))
        {
            writer.WriteLine("ply");
            writer.WriteLine("format ascii 1.0");
            writer.WriteLine($"element vertex {points.Count}");
            writer.WriteLine("property float x");
            writer.WriteLine("property float y");
            writer.WriteLine("property float z");
            writer.WriteLine("property uchar red");
            writer.WriteLine("property uchar green");
            writer.WriteLine("property uchar blue");
            writer.WriteLine("end_header");

            for (int i = 0; i < points.Count; i++)
            {
                Vector3 p = points[i];
                Color c = colors[i];
                writer.WriteLine($"{p.x} {p.y} {p.z} {(int)(c.r * 255)} {(int)(c.g * 255)} {(int)(c.b * 255)}");
            }
        }
        Debug.Log($"Exported {points.Count} textured points to {fileName}");
    }
}