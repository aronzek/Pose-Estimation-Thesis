using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class ExportContainerData : MonoBehaviour
{
    [Header("Set in Inspector")]
    public Transform containersRoot; // <- Drag your container parent GameObject here
    public string outputPath = "container_annotations.json"; // <- Relative to project root

    [System.Serializable]
    public class ContainerData
    {
        public string name;
        public Vector3 position;
        public float rotationY;
        public Vector3 size;
    }

    [System.Serializable]
    public class ContainerExportWrapper
    {
        public List<ContainerData> containers = new List<ContainerData>();
    }

    [ContextMenu("Export Container JSON")]
    public void ExportContainerMetadata()
    {
        if (containersRoot == null)
        {
            Debug.LogWarning("⚠️ Please assign the containersRoot.");
            return;
        }

        ContainerExportWrapper wrapper = new ContainerExportWrapper();

        foreach (Transform container in containersRoot)
        {
            wrapper.containers.Add(new ContainerData
            {
                name = container.name,
                position = container.position,
                rotationY = container.eulerAngles.y,
                size = container.localScale
            });
        }

        string fullPath = Path.Combine(Application.dataPath, "../", outputPath);
        File.WriteAllText(fullPath, JsonUtility.ToJson(wrapper, true));
        Debug.Log($"✅ Exported {wrapper.containers.Count} containers to:\n📄 {fullPath}");
    }
}