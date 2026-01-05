using UnityEngine;
using System.IO;
using System.Text;

public class ObjExporterWithMaterials : MonoBehaviour
{
    public Transform containersRoot;
    public string exportFolder = "ExportedContainersWithMaterials";

    [ContextMenu("Export Containers with Materials")]
    public void ExportAll()
    {
        if (containersRoot == null) return;

        string basePath = Path.Combine(Application.dataPath, "../", exportFolder);
        Directory.CreateDirectory(basePath);

        foreach (Transform container in containersRoot)
        {
            MeshFilter mf = container.GetComponent<MeshFilter>();
            MeshRenderer mr = container.GetComponent<MeshRenderer>();
            if (mf == null || mr == null) continue;

            string objName = container.name;
            string objPath = Path.Combine(basePath, objName + ".obj");
            string mtlPath = Path.Combine(basePath, objName + ".mtl");

            ExportWithMaterial(mf.sharedMesh, mr.sharedMaterial, container, objName, objPath, mtlPath);
            Debug.Log($"✅ Exported {objName} with material.");
        }
    }

    void ExportWithMaterial(Mesh mesh, Material mat, Transform tf, string name, string objPath, string mtlPath)
    {
        var sbObj = new StringBuilder();
        var sbMtl = new StringBuilder();

        string matName = mat != null ? mat.name : "DefaultMaterial";

        // OBJ Header
        sbObj.AppendLine($"mtllib {Path.GetFileName(mtlPath)}");
        sbObj.AppendLine($"o {name}");

        // Vertices
        foreach (var v in mesh.vertices)
        {
            Vector3 vtx = tf.TransformPoint(v);
            sbObj.AppendLine($"v {vtx.x} {vtx.y} {vtx.z}");
        }

        // UVs
        foreach (var uv in mesh.uv)
        {
            sbObj.AppendLine($"vt {uv.x} {uv.y}");
        }

        // Normals
        foreach (var n in mesh.normals)
        {
            Vector3 norm = tf.TransformDirection(n);
            sbObj.AppendLine($"vn {norm.x} {norm.y} {norm.z}");
        }

        sbObj.AppendLine($"usemtl {matName}");

        int[] tris = mesh.triangles;
        for (int i = 0; i < tris.Length; i += 3)
        {
            int a = tris[i] + 1;
            int b = tris[i + 1] + 1;
            int c = tris[i + 2] + 1;
            sbObj.AppendLine($"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
        }

        // Write .obj
        File.WriteAllText(objPath, sbObj.ToString());

        // Write .mtl
        sbMtl.AppendLine($"newmtl {matName}");
        sbMtl.AppendLine("Ka 1.000 1.000 1.000");
        sbMtl.AppendLine("Kd 1.000 1.000 1.000");
        sbMtl.AppendLine("Ks 0.000 0.000 0.000");
        sbMtl.AppendLine("d 1.0");
        sbMtl.AppendLine("illum 2");

        // Include texture path if available
        if (mat != null && mat.mainTexture != null)
        {
            string texName = mat.mainTexture.name + ".png";
            sbMtl.AppendLine($"map_Kd {texName}");

            // Save texture file (optional)
            string texPath = Path.Combine(Path.GetDirectoryName(mtlPath), texName);
            var tex = mat.mainTexture as Texture2D;
            if (tex != null)
            {
                byte[] bytes = tex.EncodeToPNG();
                File.WriteAllBytes(texPath, bytes);
            }
        }

        File.WriteAllText(mtlPath, sbMtl.ToString());
    }
}
