package smile.data.formula;

public class TechFactory
{
  public static Variable variable(String name)
  {
    return new Variable(name);
  }
};
