using UnityEngine;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Jobs;


public static class ComplexF {
    public const float Tau = Mathf.PI * 2f;

    public static float2 Real(float r) {
        return new float2(r, 0f);
    }

    public static float2 Imaginary(float i) {
        return new float2(0f, i);
    }

    public static float2 Mul(float2 a, float2 b) {
        return new float2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x);
    }

    public static float2 GetRotor(float freq, int samplerate) {
        float phaseStep = (Tau * freq) / samplerate;

        return new float2(
            math.cos(phaseStep),
            math.sin(phaseStep));
    }
}

public static class ComplexI {
    public static int2 Mul(int2 a, int2 b) {
        return new int2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x);
    }
}

public class ComplexInts : MonoBehaviour {
    int2 _value;

    float _lastUpdateTime;

    private void Awake() {
        _value = new int2(1, 0);
    }

    private void Update() {
        if (Time.time > _lastUpdateTime + 0.25f) {
            UpdateRotors();
            _lastUpdateTime = Time.time;
        }
        
    }

    private void UpdateRotors() {
        int2 rotor = new int2(0, 1);
        _value = ComplexI.Mul(_value, rotor);

        Debug.Log(_value);
    }

    private void OnDrawGizmos() {
        Gizmos.DrawSphere(new Vector3(_value.x, _value.y, 0f), 0.5f);
    }
}