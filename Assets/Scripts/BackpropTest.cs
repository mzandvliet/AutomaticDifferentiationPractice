using System;
using System.Collections.Generic;
using UnityEngine;

/* 
    - Build a state visualizer
    - Implement vector algebra
    - Implement other operations needed by neural nets
    - Implement using Burst

    Notes:

    - Formulating the correct job order by graph traversal and dependency checking
    for both forward and backward passes only has to be done once.

    - Don't have to calculate all possible derivatives, just the ones you need.

    - Convergence of backprop for any computational graph + config == a mandelbrot set
*/

namespace BackPropPractice {
    public class BackpropTest : MonoBehaviour {

        private void Awake() {
            var a = new ConstNode(5f, true);
            var b = new ConstNode(3f, true);
            var c = new ConstNode(2f, true);

            var add = new AddNode(a, b);
            var mul = new MultiplyNode(c, add);

            var d = new ConstNode(10f, true);
            var add2 = new AddNode(d, mul);

            var target = new ConstNode(15f, false);
            var loss = new LossNode(add2, target);

            Optimize(loss);
        }

        private static void Optimize(IFloatNode node) {
            const float rate = 0.01f;

            for (int i = 0; i < 100; i++) {
                ForwardPass(node);
                BackwardPass(node);
                ParameterUpdate(node, rate);
            }
        }

        // Traverse graph from a given node up through its inputs, in breadth-first order
        private static void TraverseInputsBF(IFloatNode node, System.Action<IFloatNode> visit) {
            var stack = new Stack<IFloatNode>();
            stack.Push(node);

            while (stack.Count > 0) {
                node = stack.Pop();

                visit(node);

                for (int i = 0; i < node.Inputs.Count; i++) {
                    stack.Push(node.Inputs[i]);
                }
            }
        }

        private static void ForwardPass(IFloatNode node) {
            // Create ordered lists of ops, burst-ready

            Debug.Log("Forward pas: " + node);

            var evalOrder = new List<IFloatNode>();
            TraverseInputsBF(node, n => evalOrder.Add(n));
            evalOrder.Reverse();

            for (int i = 0; i < evalOrder.Count; i++) {
                evalOrder[i].Forward();
                Debug.Log(i + ": " + evalOrder[i].ToString());
            }
        }

        private static void BackwardPass(IFloatNode node) {
            Debug.Log("Backwards pass: " + node);

            TraverseInputsBF(node, n => {
                float gradient = n.Output.IsConnected ? n.Output.Node.Gradients[n.Output.Inlet] : 1f;
                n.Backward(gradient);
            });

            var stack = new Stack<IFloatNode>();
            stack.Push(node);
        }

        private static void ParameterUpdate(IFloatNode node, float rate) {
            Debug.Log("Parameter Update");

            TraverseInputsBF(node, n => {
                if (n is ConstNode) {
                    var cNode = n as ConstNode;
                    if (cNode.IsLearnable) {
                        cNode.Value += n.Output.Node.Gradients[node.Output.Inlet] * rate;
                    }
                }
            });
        }
    }

    public interface IFloatNode {
        IList<IFloatNode> Inputs {
            get;
        }

        Outlet Output {
            get;
        }

        void Forward();
        void Backward(float gradient);

        float Value {
            get;
        }

        float[] Gradients {
            get;
        }

        int Id {
            get;
        }
    }

    public class Outlet {
        public IFloatNode Node {
            get;
            private set;
        }

        public int Inlet {
            get;
            private set;
        }

        public bool IsConnected {
            get { return Node != null; }
        }

        public void Connect(IFloatNode node, int inlet) {
            Node = node;
            Inlet = inlet;
        }
    }

    public abstract class AbstractFloatNode : IFloatNode {
        public IList<IFloatNode> Inputs {
            get;
            private set;
        }

        public Outlet Output {
            get;
        }

        public float Value {
            get;
            set;
        }

        public float[] Gradients {
            get;
            private set;
        }

        public int Id {
            get;
            private set;
        }

        private static int Count;

        public AbstractFloatNode(int inlets) {
            Gradients = new float[inlets];
            Inputs = new IFloatNode[inlets];
            Output = new Outlet();
            Id = Count++;
        }

        public abstract void Forward();
        public abstract void Backward(float gradient);
    }

    public class ConstNode : AbstractFloatNode {
        public bool IsLearnable {
            get;
            private set;
        }

        public ConstNode(float value, bool learnable) : base(0) {
            Value = value;
            IsLearnable = learnable;
        }

        public override void Forward() { }
        public override void Backward(float gradient) {
        }

        public override string ToString() {
            return "ID: " + Id + ", Const: " + Value;
        }
    }

    public class AddNode : AbstractFloatNode {
        public AddNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[0].Value + Inputs[1].Value;
        }

        public override void Backward(float gradient) {
            Gradients[0] = gradient;
            Gradients[1] = gradient;
        }

        public override string ToString() {
            return "ID: " + Id + ", + " + Value;
        }
    }

    public class MultiplyNode : AbstractFloatNode {
        public MultiplyNode(IFloatNode a, IFloatNode b) : base(2) {
            Inputs[0] = a;
            Inputs[1] = b;

            a.Output.Connect(this, 0);
            b.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[0].Value * Inputs[1].Value;
        }

        public override void Backward(float gradient) {
            Gradients[0] = gradient * Inputs[1].Value;
            Gradients[1] = gradient * Inputs[0].Value;
        }

        public override string ToString() {
            return "ID: " + Id + ", * " + Value;
        }
    }

    public class LossNode : AbstractFloatNode {
        public LossNode(IFloatNode result, IFloatNode target) : base(2) {
            Inputs[0] = result;
            Inputs[1] = target;

            result.Output.Connect(this, 0);
            target.Output.Connect(this, 1);
        }

        public override void Forward() {
            Value = Inputs[1].Value - Inputs[0].Value;
        }

        public override void Backward(float gradient) {
            // Note: this starts the gradient, since it is always the last node
            Gradients[0] = Value;
            Gradients[1] = Value;
        }

        public override string ToString() {
            return "ID: " + Id + ", Loss: " + Value;
        }
    }

    [AttributeUsage(
    AttributeTargets.Class |
    AttributeTargets.Field |
    AttributeTargets.Property)]
    public class OptimizeParameterAttribute : Attribute {

    }
}
