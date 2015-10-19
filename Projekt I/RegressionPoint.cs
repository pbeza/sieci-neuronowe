namespace sieci_neuronowe
{
    public struct RegressionPoint
    {
        public double X;
        public double Y;
        public double CorrectY;

        public RegressionPoint(double x, double y, double correctY)
        {
            this.X = x;
            this.Y = y;
            this.CorrectY = correctY;
        }
    }
}