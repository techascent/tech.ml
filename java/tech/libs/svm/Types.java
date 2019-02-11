package tech.libs.svm;


import com.sun.jna.*;
import java.util.*;


public interface Types extends Library {

  public static interface PrintString extends Callback {
    void Print(String data);
  }

  public static class SVMNode extends Structure {
    public int index;
    public double value;

    public static class ByReference extends SVMNode implements Structure.ByReference {}
    public static class ByValue extends SVMNode implements Structure.ByValue {}
    public SVMNode () {}
    public SVMNode (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() {
      return Arrays.asList(new String[]
	{ "index", "value" });
    }
  }

  public static class SVMProblem extends Structure {
    public int l;
    public Pointer y;
    public Pointer x;

    public static class ByReference extends SVMProblem implements Structure.ByReference {}
    public static class ByValue extends SVMProblem implements Structure.ByValue {}
    public SVMProblem () {}
    public SVMProblem (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() {
      return Arrays.asList(new String[]
	{ "l", "y", "x" });
    }
  }

  public static class SVMParameter extends Structure {
    public int svm_type;
    public int kernel_type;
    public int degree;	/* for poly */
    public double gamma;	/* for poly/rbf/sigmoid */
    public double coef0;	/* for poly/sigmoid */

    /* these are for training only */
    public double cache_size; /* in MB */
    public double eps;	/* stopping criteria */
    public double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
    public int nr_weight;		/* for C_SVC */
    public Pointer weight_label;	/* for C_SVC */
    public Pointer weight;		/* for C_SVC */
    public double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
    public double p;	/* for EPSILON_SVR */
    public int shrinking;	/* use the shrinking heuristics */
    public int probability; /* do probability estimates */

    public static class ByReference extends SVMParameter implements Structure.ByReference {}
    public static class ByValue extends SVMParameter implements Structure.ByValue {}
    public SVMParameter () {}
    public SVMParameter (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() {
      return Arrays.asList(new String[]
	{
	  "svm_type",
	  "kernel_type",
	  "degree",
	  "gamma",
	  "coef0",
	  "cache_size",
	  "eps",
	  "C",
	  "nr_weight",
	  "weight_label",
	  "weight",
	  "nu",
	  "p",
	  "shrinking",
	  "probability"
	});
    }
  }

  public static class SVMModel extends Structure {
    public SVMParameter param;	        /* parameter */
    public int nr_class;		/* number of classes, = 2 in regression/one class svm */
    public int l;			/* total #SV */
    public Pointer SV;                 /* struct svm_node **SV; SVs (SV[l]) */
    public Pointer sv_coef;	        /* double** sv_coef coefficients for SVs in decision functions (sv_coef[k-1][l]) */
    public Pointer rho;		/* double* constants in decision functions (rho[k*(k-1)/2]) */
    public Pointer probA;		/* double* pariwise probability information */
    public Pointer probB;              /* double* */
    public Pointer sv_indices;         /* int* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

    /* for classification only */

    public Pointer label;		/* int* label of each class (label[k]) */
    public Pointer nSV;		/* int* number of SVs for each class (nSV[k]) */
    /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    /* XXX */
    public int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */

    public static class ByReference extends SVMModel implements Structure.ByReference {}
    public static class ByValue extends SVMModel implements Structure.ByValue {}
    public SVMModel () {}
    public SVMModel (Pointer p ) { super(p); read(); }
    protected List getFieldOrder() {
      return Arrays.asList(new String[]
	{
	  "param",
	  "nr_class",
	  "l",
	  "SV",
	  "sv_coef",
	  "rho",
	  "probA",
	  "probB",
	  "sv_indices",
	  "label",
	  "nSV",
	  "free_sv",
	});
    }
  }
}
