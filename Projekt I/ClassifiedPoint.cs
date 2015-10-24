namespace sieci_neuronowe
{
    public struct ClassifiedPoint
    {
        public int Category;
        public int Correct;
        public double X;
        public double Y;

        public ClassifiedPoint(double x, double y, int category, int correct)
        {
            X = x;
            Y = y;
            Category = category;
            Correct = correct;
        }
    }
}
