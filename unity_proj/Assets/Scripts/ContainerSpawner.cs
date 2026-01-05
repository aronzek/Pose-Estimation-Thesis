using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class ContainerSpawner : MonoBehaviour
{
    [Header("Container Settings")]
    public GameObject containerPrefab;
    public Transform containerParent;
    public int containerCount = 60;
    public float spacing = 9f;
    public string outputFileName = "unity_containers.json";

    private List<ContainerInfo> containerInfos = new List<ContainerInfo>();

    [ContextMenu("Generate Containers")]
    public void GenerateContainers()
    {
        containerInfos.Clear();

        // Clear existing containers under parent
        if (containerParent != null)
        {
            for (int i = containerParent.childCount - 1; i >= 0; i--)
            {
                DestroyImmediate(containerParent.GetChild(i).gameObject);
            }
        }

        List<Vector3> usedPositions = new List<Vector3>();

        // Get ground dimensions from this GameObject's transform
        Vector3 groundPos = transform.position;
        Vector3 groundSize = transform.localScale * 10f; // Convert scale to meters

        int placed = 0;
        int attempts = 0;
        int maxAttempts = containerCount * 20;

        while (placed < containerCount && attempts < maxAttempts)
        {
            float halfW = groundSize.x / 2f;
            float halfD = groundSize.z / 2f;

            float randX = Random.Range(groundPos.x - halfW + spacing, groundPos.x + halfW - spacing);
            float randZ = Random.Range(groundPos.z - halfD + spacing, groundPos.z + halfD - spacing);
            float rotY = Random.Range(0f, 180f);

            float containerHeight = 2.8f; // Adjust based on your model’s actual height
            Vector3 candidate = new Vector3(randX, groundPos.y + containerHeight / 2f, randZ);
            bool overlaps = false;

            foreach (Vector3 pos in usedPositions)
            {
                if (Vector3.Distance(pos, candidate) < spacing)
                {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps)
            {
                GameObject container = Instantiate(containerPrefab, candidate, Quaternion.Euler(0, rotY, 0), containerParent);
                Vector3 size = containerPrefab.transform.localScale * 10f; // Adjust if needed

                containerInfos.Add(new ContainerInfo(candidate, size, rotY));
                usedPositions.Add(candidate);
                placed++;
            }

            attempts++;
        }

        Debug.Log($"[ContainerSpawner] Spawned {placed} containers with spacing {spacing} across ground {groundSize.x} x {groundSize.z}.");
    }

    [ContextMenu("Save Container JSON")]
    public void SaveToJson()
    {
        string json = JsonHelper.ToJson(containerInfos.ToArray(), true);
        string path = Path.Combine(Application.dataPath, "..", "SyntheticExports", outputFileName);

        File.WriteAllText(path, json);
        Debug.Log($"[ContainerSpawner] Saved container config to: {path}");
    }
}
