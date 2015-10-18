namespace sieci_neuronowe
{
    public struct NeuroPoint
    {
        public int Category;
        public int Correct;
        public double X;
        public double Y;

        public NeuroPoint(double x, double y, int category, int correct)
        {
            X = x;
            Y = y;
            Category = category;
            Correct = correct;
        }
    }
}
