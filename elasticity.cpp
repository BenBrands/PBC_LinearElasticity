#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/manifold.h>

#include <fstream>
#include <iostream>
//#include <filesystem>


namespace Elasticity
{
  using namespace dealii;


  template <int dim>
  SymmetricTensor<4,dim> get_elasticity_tensor(const double lambda, const double mu)
  {
  	SymmetricTensor<4,dim> temp;
  	for (unsigned int i=0; i<dim; ++i)
  		for (unsigned int j=0; j<dim; ++j)
  			for (unsigned int k=0; k<dim; ++k)
  				for (unsigned int l=0; l<dim; ++l)
  					temp[i][j][k][l] = ( ((i==j)&&(k==l)) ? lambda : 0 ) + ( ((i==k)&&(j==l)) ? mu : 0 ) + ( ((i==l)&&(j==k)) ? mu : 0 );

  	return temp;
  }


  template <int dim>
  class MakroField : public Function<dim>
  {
	public:

	  //Constructor for number of solution variables
	  MakroField(const SymmetricTensor<2,dim> &strain_tensor);

	  virtual ~MakroField() {}

	  double value(const Point<dim> &p,
			  	   const unsigned int component=0) const;

	  void value_list(const std::vector<Point<dim>> &point_list,
			  	  	  std::vector<double> &value_list,
					  const unsigned int component=0) const;

	  void vector_value(const Point<dim> &p,
			  	  	  	Vector<double> &value) const;

	  void vector_value_list(const std::vector<Point<dim>> &point_list,
			  	  	  	  	 std::vector<Vector<double>> &value_list) const;

	  const SymmetricTensor<2,dim> epsilon;
  };

  template <int dim>
  MakroField<dim>::MakroField(const SymmetricTensor<2,dim> &strain_tensor)
  :
  Function<dim>(dim+dim),
  epsilon(strain_tensor)
  {}



  template <int dim>
  void
  MakroField<dim>::vector_value(const Point<dim> &p,
		  	  	  	  	  	  	Vector<double> &value) const
  {
	  Assert(value.size()==2*dim,ExcDimensionMismatch(value.size(),2*dim));

	  Tensor<1,dim> temp, x(p);
	  temp = epsilon*x;

	  for (unsigned int i=0; i<dim; ++i)
		  value[i]=temp[i];
  }

  template <int dim>
  void
  MakroField<dim>::vector_value_list(const std::vector<Point<dim>> &p_list,
 	  	  	 	 	 	 	 	 	 std::vector<Vector<double>> &v_list) const
  {
	  Assert(v_list.size()==p_list.size(),ExcDimensionMismatch(v_list.size(),p_list.size()));

	  for (unsigned int i=0; i<p_list.size(); ++i)
		  MakroField<dim>::vector_value(p_list[i],v_list[i]);
  }

  template <int dim>
  double
  MakroField<dim>::value(const Point<dim> &p,
	  	   	   	   		 const unsigned int component) const
  {
	  Assert(component>0 && component<2*dim,ExcIndexRange(component,0,2*dim+1));

	  Tensor<1,dim> temp, x(p);
	  temp = epsilon*x;

	  if (component<dim)
		  return temp[component];
	  else
		  return 0.;
  }

  template <int dim>
  void
  MakroField<dim>::value_list(const std::vector<Point<dim>> &p_list,
  	  	  	  	  	  	  	  std::vector<double> &v_list,
							  const unsigned int component) const
  {
	  Assert(v_list.size()==p_list.size(),ExcDimensionMismatch(v_list.size(),p_list.size()));

	  for (unsigned int i=0; i<p_list.size(); ++i)
		  v_list[i] = MakroField<dim>::value(p_list[i],component);
  }




  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(const unsigned int fe_degree,
    			   const std::string &filename_mesh);

    ~ElasticProblem();

    void run_I(const SymmetricTensor<2,dim> &epsilon);

    void run_II(const SymmetricTensor<2,dim> &epsilon);

    void run_III(const SymmetricTensor<2,dim> &epsilon);

  private:

    void setup_system_1();

    void setup_system_2();

    void solve(BlockSparseMatrix<double> &matrix,
    		   const BlockVector<double> &rhs,
			   BlockVector<double> &solution) const;

    void assemble_system_lDBC(BlockSparseMatrix<double> &matrix,
    						  const SymmetricTensor<2,dim> &epsilon,
							  BlockVector<double> &rhs) const;

    void assemble_system_pDBC_1(BlockSparseMatrix<double> &matrix,
    						    const SymmetricTensor<2,dim> &epsilon,
							    BlockVector<double> &rhs) const;

    void assemble_system_pDBC_2(BlockSparseMatrix<double> &matrix,
			  	  	  	  	  	const SymmetricTensor<2,dim> &epsilon,
								BlockVector<double> &rhs) const;

    SymmetricTensor<2,dim> averaged_sigma(const BlockVector<double> &solution,
    									  const SymmetricTensor<2,dim> &epsilon);

    void output(const BlockVector<double> &solution,
    			const std::string &file_name,
				const SymmetricTensor<2,dim> &epsilon) const;

    void refine();

    const std::string						mesh_file;
    Triangulation<dim>   					triangulation;
	FESystem<dim> 							fe_1, fe_2;
	FEValuesExtractors::Vector 				displacements, multipliers;
	hp::FECollection<dim>					fe_collection;
	hp::DoFHandler<dim> 					dof_handler;
	AffineConstraints<double>				constraints;

	hp::QCollection<dim>					quadrature_collection;
	hp::QCollection<dim-1> 					face_quadrature_collection;

    BlockSparsityPattern				    sparsity_pattern;
    std::vector<types::global_dof_index>	dofs_per_block, relevant_dofs_per_block;

    const double		 					lambda_matrix, mu_matrix;
    const double		 					lambda_inclusion, mu_inclusion;

    SymmetricTensor<4,dim> 					c4_matrix, c4_inclusion;
  };


  template <int dim>
  ElasticProblem<dim>::ElasticProblem(const unsigned int fe_degree,
		  	  	  	  	  	  	  	  const std::string &filename_mesh)
  :
	mesh_file(filename_mesh),
  	dof_handler(triangulation),
	fe_1(FESystem<dim>(FE_Q<dim>(fe_degree),dim),1,FESystem<dim>(FE_Q<dim>(fe_degree),dim),1),
	fe_2(FESystem<dim>(FE_Q<dim>(fe_degree),dim),1,FESystem<dim>(FE_Nothing<dim>(),dim),1),
	displacements(0),multipliers(dim),
	lambda_matrix(12.),mu_matrix(8.),
	lambda_inclusion(120.),mu_inclusion(80.)
  {
	fe_collection.push_back(fe_1);
	fe_collection.push_back(fe_2);

	quadrature_collection.push_back(QGauss<dim>(2*fe_degree));
	quadrature_collection.push_back(QGauss<dim>(2*fe_degree));

	face_quadrature_collection.push_back(QGauss<dim-1>(2*fe_degree));
	face_quadrature_collection.push_back(QGauss<dim-1>(2*fe_degree));

	c4_matrix = get_elasticity_tensor<dim>(lambda_matrix,mu_matrix);
	c4_inclusion = get_elasticity_tensor<dim>(lambda_inclusion,mu_inclusion);
  }


  template <int dim>
  ElasticProblem<dim>::~ElasticProblem()
  {
    dof_handler.clear ();
  }


  template <int dim>
  void ElasticProblem<dim>::setup_system_1()
  {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input(mesh_file.c_str());
	grid_in.read_ucd(input);

	for (typename Triangulation<dim>::active_face_iterator face = triangulation.begin_active_face(); face != triangulation.end_face(); ++face)
		if (face->at_boundary())
		{
			if ( std::abs(face->center()[0] - 1) < 1e-6 || std::abs(face->center()[1] - 1) < 1e-6)
				face->set_boundary_id(1);
			else
				face->set_boundary_id(0);
		}
	for (typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
	  if (cell->at_boundary())
	    cell->set_active_fe_index(0);
	  else
	    cell->set_active_fe_index(1);

	dof_handler.distribute_dofs(fe_collection);
	DoFRenumbering::Cuthill_McKee(dof_handler);
	DoFRenumbering::block_wise(dof_handler);

	dofs_per_block.resize(fe_collection.n_blocks());
	DoFTools::count_dofs_per_block(dof_handler,dofs_per_block);

	FEValuesExtractors::Vector displacements(0), multipliers(dim);
	ComponentMask displacement_mask = fe_collection.component_mask(displacements);
	ComponentMask multiplier_mask = fe_collection.component_mask(multipliers);
	std::vector<bool> boundary_multiplier_dofs(dof_handler.n_dofs(),false);
	DoFTools::extract_dofs_with_support_on_boundary(dof_handler,multiplier_mask,boundary_multiplier_dofs);
	std::vector<bool> interior_multiplier_dofs(dof_handler.n_dofs(),false);
	for (unsigned int i=0; i<dofs_per_block[1]; ++i)
	  if (!boundary_multiplier_dofs[ dofs_per_block[0]+i ])
	    interior_multiplier_dofs[ dofs_per_block[0]+i ] = true;
	DoFRenumbering::sort_selected_dofs_back(dof_handler,interior_multiplier_dofs);

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler,constraints);
	constraints.close ();

	BlockDynamicSparsityPattern csp_displacement(fe_collection.n_blocks(),fe_collection.n_blocks());
	for (unsigned int i=0; i<fe_collection.n_blocks(); ++i)
	  for (unsigned int j=0; j<fe_collection.n_blocks(); ++j)
	    csp_displacement.block(i,j).reinit(dofs_per_block[i], dofs_per_block[j]);

	csp_displacement.collect_sizes();
	Table<2, DoFTools::Coupling> tmp_coupling(fe_collection.n_components(),fe_collection.n_components());
	for (unsigned int i=0; i<fe_collection.n_components(); ++i)
	  for (unsigned int j=0; j<fe_collection.n_components(); ++j)
	    if (i<dim && j<dim)
		  tmp_coupling[i][j] = DoFTools::always;
		else
		  tmp_coupling[i][j] = DoFTools::none;
	DoFTools::make_sparsity_pattern(dof_handler,tmp_coupling,csp_displacement,constraints,false);

	unsigned int n_boundary_multiplier_dofs = 0;
	for (unsigned int i=0; i<boundary_multiplier_dofs.size(); ++i)
		if (boundary_multiplier_dofs[i])
			++n_boundary_multiplier_dofs;
	relevant_dofs_per_block.resize(fe_collection.n_blocks());
	relevant_dofs_per_block[0] = dofs_per_block[0];
	relevant_dofs_per_block[1] = n_boundary_multiplier_dofs;

	BlockDynamicSparsityPattern csp_multiplier(fe_collection.n_blocks(),fe_collection.n_blocks());
	for (unsigned int i=0; i<fe_collection.n_blocks(); ++i)
		for (unsigned int j=0; j<fe_collection.n_blocks(); ++j)
			csp_multiplier.block(i,j).reinit(relevant_dofs_per_block[i],relevant_dofs_per_block[j]);
	csp_multiplier.collect_sizes();

	unsigned int cell_num=0;
	for (typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell != dof_handler.end(); ++cell, ++cell_num)
	  if (cell->at_boundary())
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
		  if (cell->face(face)->at_boundary())
		  {
		    const unsigned int dofs_per_face = cell->get_fe().dofs_per_face;
			std::vector<types::global_dof_index> face_indices (dofs_per_face);
			cell->face(face)->get_dof_indices (face_indices,0);
			std::set<types::global_dof_index> displacement_face_dofs, multiplier_face_dofs;
			for (unsigned int i=0; i<face_indices.size(); ++i)
			{
			  if (face_indices[i] < relevant_dofs_per_block[0])
			    displacement_face_dofs.insert (face_indices[i]);
			  else
				multiplier_face_dofs.insert (face_indices[i]);
			}
			for (std::set<types::global_dof_index>::const_iterator it_d=displacement_face_dofs.begin(); it_d != displacement_face_dofs.end(); ++it_d)
			  for (std::set<types::global_dof_index>::const_iterator it_m=multiplier_face_dofs.begin(); it_m != multiplier_face_dofs.end(); ++it_m)
			  {
			    csp_multiplier.add(*it_d,*it_m);
				csp_multiplier.add(*it_m,*it_d);
			  }
		  }
	sparsity_pattern.reinit(fe_collection.n_blocks(),fe_collection.n_blocks());
	sparsity_pattern.block(0,0).copy_from(csp_displacement.block(0,0));
	sparsity_pattern.block(1,1).copy_from(csp_multiplier.block(1,1));
	sparsity_pattern.block(0,1).copy_from(csp_multiplier.block(0,1));
	sparsity_pattern.block(1,0).copy_from(csp_multiplier.block(1,0));
	sparsity_pattern.collect_sizes();
	sparsity_pattern.compress();

	std::ofstream out("sparsity_pattern_1.gp");
	sparsity_pattern.print_gnuplot(out);
  }


  template <int dim>
  void ElasticProblem<dim>::setup_system_2()
  {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input(mesh_file.c_str());
	grid_in.read_ucd(input);

	for (typename Triangulation<dim>::active_face_iterator face = triangulation.begin_active_face(); face != triangulation.end_face(); ++face)
		if (face->at_boundary())
		{
			if ( std::abs(face->center()[0] - 1) < 1e-6 || std::abs(face->center()[1] - 1) < 1e-6)
				face->set_boundary_id(1);
			else
				face->set_boundary_id(0);
		}
	for (typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
	  if (cell->at_boundary())
	    cell->set_active_fe_index(0);
	  else
	    cell->set_active_fe_index(1);

	dof_handler.distribute_dofs(fe_collection);
	DoFRenumbering::Cuthill_McKee(dof_handler);
	DoFRenumbering::block_wise(dof_handler);

	dofs_per_block.resize(fe_collection.n_blocks());
	DoFTools::count_dofs_per_block(dof_handler,dofs_per_block);

	FEValuesExtractors::Vector displacements(0), multipliers(dim);
	ComponentMask displacement_mask = fe_collection.component_mask(displacements);
	ComponentMask multiplier_mask = fe_collection.component_mask(multipliers);
	std::vector<bool> boundary_multiplier_dofs(dof_handler.n_dofs(),false);
	DoFTools::extract_dofs_with_support_on_boundary(dof_handler,multiplier_mask,boundary_multiplier_dofs);
	std::vector<bool> interior_multiplier_dofs(dof_handler.n_dofs(),false);
	for (unsigned int i=0; i<dofs_per_block[1]; ++i)
	  if (!boundary_multiplier_dofs[ dofs_per_block[0]+i ])
	    interior_multiplier_dofs[ dofs_per_block[0]+i ] = true;
	DoFRenumbering::sort_selected_dofs_back(dof_handler,interior_multiplier_dofs);

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler,constraints);
	constraints.close ();

	unsigned int n_boundary_multiplier_dofs = 0;
	for (unsigned int i=0; i<boundary_multiplier_dofs.size(); ++i)
		if (boundary_multiplier_dofs[i])
			++n_boundary_multiplier_dofs;
	relevant_dofs_per_block.resize(fe_collection.n_blocks()+1);
	relevant_dofs_per_block[0] = dofs_per_block[0];
	relevant_dofs_per_block[1] = n_boundary_multiplier_dofs;
	relevant_dofs_per_block[2] = dim;

	BlockDynamicSparsityPattern csp_displacement(fe_collection.n_blocks(),fe_collection.n_blocks());
	for (unsigned int i=0; i<fe_collection.n_blocks(); ++i)
	  for (unsigned int j=0; j<fe_collection.n_blocks(); ++j)
	    csp_displacement.block(i,j).reinit(dofs_per_block[i], dofs_per_block[j]);

	csp_displacement.collect_sizes();
	Table<2, DoFTools::Coupling> tmp_coupling(fe_collection.n_components(),fe_collection.n_components());
	for (unsigned int i=0; i<fe_collection.n_components(); ++i)
	  for (unsigned int j=0; j<fe_collection.n_components(); ++j)
	    if (i<dim && j<dim)
		  tmp_coupling[i][j] = DoFTools::always;
		else
		  tmp_coupling[i][j] = DoFTools::none;
	DoFTools::make_sparsity_pattern(dof_handler,tmp_coupling,csp_displacement,constraints,false);


	BlockDynamicSparsityPattern csp_multiplier(relevant_dofs_per_block.size(),relevant_dofs_per_block.size());
	for (unsigned int i=0; i<relevant_dofs_per_block.size(); ++i)
		for (unsigned int j=0; j<relevant_dofs_per_block.size(); ++j)
			csp_multiplier.block(i,j).reinit(relevant_dofs_per_block[i],relevant_dofs_per_block[j]);
	csp_multiplier.collect_sizes();

	unsigned int cell_num=0;
	for (typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(); cell != dof_handler.end(); ++cell, ++cell_num)
	  if (cell->at_boundary())
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
		  if (cell->face(face)->at_boundary())
		  {
		    const unsigned int dofs_per_face = cell->get_fe().dofs_per_face;
			std::vector<types::global_dof_index> face_indices (dofs_per_face);
			cell->face(face)->get_dof_indices (face_indices,0);
			std::set<types::global_dof_index> displacement_face_dofs, multiplier_face_dofs;
			for (unsigned int i=0; i<face_indices.size(); ++i)
			{
			  if (face_indices[i] < relevant_dofs_per_block[0])
			    displacement_face_dofs.insert (face_indices[i]);
			  else
				multiplier_face_dofs.insert (face_indices[i]);
			}
			for (std::set<types::global_dof_index>::const_iterator it_d=displacement_face_dofs.begin(); it_d != displacement_face_dofs.end(); ++it_d)
			  for (std::set<types::global_dof_index>::const_iterator it_m=multiplier_face_dofs.begin(); it_m != multiplier_face_dofs.end(); ++it_m)
			  {
			    csp_multiplier.add(*it_d,*it_m);
				csp_multiplier.add(*it_m,*it_d);
			  }
		  }
	std::vector<types::global_dof_index> relevant_offsets(relevant_dofs_per_block.size(),0);
	relevant_offsets[1] = relevant_dofs_per_block[0];
	relevant_offsets[2] = relevant_offsets[1]+relevant_dofs_per_block[1];

	for (unsigned int i=0; i<relevant_dofs_per_block[0]; ++i)
	{
		for (unsigned int j=0; j<relevant_dofs_per_block[2]; ++j)
		{
			csp_multiplier.add(i,j+relevant_offsets[2]);
			csp_multiplier.add(j+relevant_offsets[2],i);
		}
	}
	sparsity_pattern.reinit(fe_collection.n_blocks()+1,fe_collection.n_blocks()+1);
	sparsity_pattern.block(0,0).copy_from(csp_displacement.block(0,0));
	sparsity_pattern.block(0,1).copy_from(csp_multiplier.block(0,1));
	sparsity_pattern.block(0,2).copy_from(csp_multiplier.block(0,2));
	sparsity_pattern.block(1,0).copy_from(csp_multiplier.block(1,0));
	sparsity_pattern.block(1,1).copy_from(csp_multiplier.block(1,1));
	sparsity_pattern.block(1,2).copy_from(csp_multiplier.block(1,2));
	sparsity_pattern.block(2,0).copy_from(csp_multiplier.block(2,0));
	sparsity_pattern.block(2,1).copy_from(csp_multiplier.block(2,1));
	sparsity_pattern.block(2,2).copy_from(csp_multiplier.block(2,2));
	sparsity_pattern.collect_sizes();
	sparsity_pattern.compress();

	std::ofstream out("sparsity_pattern_2.gp");
	sparsity_pattern.print_gnuplot(out);
  }


  template <int dim>
  void ElasticProblem<dim>::assemble_system_lDBC(BlockSparseMatrix<double> &matrix,
  						  	  	  	  	  	  	 const SymmetricTensor<2,dim> &epsilon,
												 BlockVector<double> &rhs) const
  {
	 BlockVector<double> temp_solution(dofs_per_block);
     matrix = 0;
	 rhs = 0;

	 const double reciprocal_vol = 1 / GridTools::volume(triangulation);

	 hp::FEValues<dim> hp_fe_v(fe_collection,quadrature_collection,
			 	 	 	 	   update_values | update_quadrature_points | update_JxW_values | update_gradients);
	 hp::FEFaceValues<dim> hp_fe_face_v(fe_collection,face_quadrature_collection,
			 	 	 	 	 	 	 	update_values | update_quadrature_points | update_JxW_values | update_gradients);

 	 std::vector<dealii::types::global_dof_index> local_dof_indices;

	 unsigned int dofs_per_cell_0 = fe_collection[0].dofs_per_cell;
	 unsigned int dofs_per_cell_1 = fe_collection[1].dofs_per_cell;

	 FullMatrix<double> cell_matrix(dofs_per_cell_0+dofs_per_cell_1,dofs_per_cell_0+dofs_per_cell_1);
	 Vector<double> cell_rhs(dofs_per_cell_0+dofs_per_cell_1);

	 typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	 typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	 for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
	 {
	   cell_matrix = 0;
	   cell_rhs = 0;

	   hp_fe_v.reinit (cell);
	   const FEValues<dim> &fe_v = hp_fe_v.get_present_fe_values();

	   unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	   local_dof_indices.resize(dofs_per_cell);
	   cell->get_dof_indices(local_dof_indices);
	   cell_matrix.reinit(local_dof_indices.size(),local_dof_indices.size());
	   cell_rhs.reinit(local_dof_indices.size());

	   std::vector<SymmetricTensor<2,dim>> tensor_eps(fe_v.n_quadrature_points), tensors_sigma(fe_v.n_quadrature_points);
	   fe_v[displacements].get_function_symmetric_gradients(temp_solution,tensor_eps);

	   unsigned int material_id = cell->material_id();
	   SymmetricTensor<4,dim> c_4;
	   if (material_id==2)
	     c_4 = c4_matrix;
	   else if (material_id==1)
		 c_4 = c4_inclusion;
	   else
		 AssertThrow(false,ExcMessage("invalid material id"));

	   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
	   {
		  tensor_eps[q_p] += epsilon;
		  tensors_sigma[q_p] = c_4 * tensor_eps[q_p];

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		     SymmetricTensor<2,dim> sym_grad_N_i = fe_v[displacements].symmetric_gradient(i,q_p);

		     cell_rhs[i] += sym_grad_N_i * tensor_eps[q_p] * fe_v.JxW(q_p);

		     for (unsigned int j=0; j<dofs_per_cell; ++j)
		     {
		    	SymmetricTensor<2,dim> sym_grad_N_j = fe_v[displacements].symmetric_gradient(j,q_p);
		    	SymmetricTensor<2,dim> sigma_j = c_4 * sym_grad_N_j;

		    	cell_matrix(i,j) += (sym_grad_N_i * sigma_j) * fe_v.JxW(q_p);
		     }
		  }
	   }
	   if (cell->at_boundary())
	   {
		  for (unsigned int face_num=0; face_num < GeometryInfo<dim>::faces_per_cell; ++face_num)
		  {
			 if ( cell->face(face_num)->at_boundary() )
			 {
			    hp_fe_face_v.reinit(cell,face_num);
			    const FEFaceValues<dim> &fe_face_v = hp_fe_face_v.get_present_fe_values();
			    const Quadrature<dim-1> &face_quadrature = hp_fe_face_v.get_present_fe_values().get_quadrature();
			    std::vector<Point<dim>> f_q_points(face_quadrature.size());
			    f_q_points = fe_face_v.get_quadrature_points();

			    Assert (dofs_per_cell == dofs_per_cell_0, ExcInternalError());

			    std::vector<Tensor<1,dim>> tensors_disp(face_quadrature.size()), tensors_lambda(face_quadrature.size());
			    fe_face_v[displacements].get_function_values(temp_solution,tensors_disp);
			    fe_face_v[multipliers].get_function_values(temp_solution,tensors_lambda);

			    for (unsigned int f_qp=0; f_qp<f_q_points.size(); ++f_qp)
			    {
			       for (unsigned int i=0; i<dofs_per_cell; ++i)
			       {
			    	  Tensor<1,dim> n_i_u = fe_face_v[displacements].value(i,f_qp);
			    	  Tensor<1,dim> n_i_lambda = fe_face_v[multipliers].value(i,f_qp);

			    	  cell_rhs[i] += reciprocal_vol * (n_i_lambda * tensors_disp[f_qp]) * fe_face_v.JxW(f_qp);
			    	  cell_rhs[i] -= reciprocal_vol * (n_i_u * tensors_lambda[f_qp]) * fe_face_v.JxW(f_qp);

			    	  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    	  {
			    		 Tensor<1,dim> n_j_u = fe_face_v[displacements].value(j,f_qp);
			    		 Tensor<1,dim> n_j_lambda = fe_face_v[multipliers].value(j,f_qp);

			    		 cell_matrix(i,j) += reciprocal_vol * (n_i_u * n_j_lambda) * fe_face_v.JxW(f_qp);
			    		 cell_matrix(i,j) += reciprocal_vol * (n_i_lambda * n_j_u) * fe_face_v.JxW(f_qp);
			    	  }
			       }
			    }
			 }
		  }
	   }
	   for (unsigned int i=0; i<dofs_per_cell; ++i)
	   {
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
			  if (local_dof_indices[i] < matrix.m() && local_dof_indices[j] < matrix.n())
				 matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
		  }
		  if (local_dof_indices[i] < rhs.size())
		    rhs[local_dof_indices[i]] += cell_rhs[i];
	   }
	}
  }


  template <int dim>
  void ElasticProblem<dim>::assemble_system_pDBC_1(BlockSparseMatrix<double> &matrix,
  						  	  	  	  	  	  	   const SymmetricTensor<2,dim> &epsilon,
												   BlockVector<double> &rhs) const
  {
	 BlockVector<double> temp_solution(dofs_per_block);
     matrix = 0;
	 rhs = 0;

	 const double reciprocal_vol = 1 / GridTools::volume(triangulation);

	 hp::FEValues<dim> hp_fe_v(fe_collection,quadrature_collection,
			 	 	 	 	   update_values | update_quadrature_points | update_JxW_values | update_gradients);
	 hp::FEFaceValues<dim> hp_fe_face_v(fe_collection,face_quadrature_collection,
			 	 	 	 	 	 	 	update_values | update_quadrature_points | update_JxW_values | update_gradients);

 	 std::vector<dealii::types::global_dof_index> local_dof_indices;

	 unsigned int dofs_per_cell_0 = fe_collection[0].dofs_per_cell;
	 unsigned int dofs_per_cell_1 = fe_collection[1].dofs_per_cell;

	 FullMatrix<double> cell_matrix;
	 Vector<double> cell_rhs;

	 typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	 typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	 for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
	 {
	   hp_fe_v.reinit (cell);
	   const FEValues<dim> &fe_v = hp_fe_v.get_present_fe_values();

	   unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	   local_dof_indices.resize(dofs_per_cell);
	   cell->get_dof_indices(local_dof_indices);
	   cell_matrix.reinit(local_dof_indices.size(),local_dof_indices.size());
	   cell_rhs.reinit(local_dof_indices.size());

	   cell_matrix = 0;
	   cell_rhs = 0;

	   std::vector<SymmetricTensor<2,dim>> tensor_eps(fe_v.n_quadrature_points), tensors_sigma(fe_v.n_quadrature_points);
	   fe_v[displacements].get_function_symmetric_gradients(temp_solution,tensor_eps);
	   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
		   tensor_eps[q_p] += epsilon;

	   unsigned int material_id = cell->material_id();
	   SymmetricTensor<4,dim> c_4;
	   if (material_id==2)
	     c_4 = c4_matrix;
	   else if (material_id==1)
		 c_4 = c4_inclusion;
	   else
		 AssertThrow(false,ExcMessage("invalid material id"));

	   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
	   {
		  tensors_sigma[q_p] = c_4 * tensor_eps[q_p];

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		     SymmetricTensor<2,dim> sym_grad_N_i = fe_v[displacements].symmetric_gradient(i,q_p);

		     cell_rhs[i] -= sym_grad_N_i * tensors_sigma[q_p] * fe_v.JxW(q_p);

		     for (unsigned int j=0; j<dofs_per_cell; ++j)
		     {
		    	SymmetricTensor<2,dim> sym_grad_N_j = fe_v[displacements].symmetric_gradient(j,q_p);
		    	SymmetricTensor<2,dim> sigma_j = c_4 * sym_grad_N_j;

		    	cell_matrix(i,j) += (sym_grad_N_i * sigma_j) * fe_v.JxW(q_p);
		     }
		  }
	   }
	   if (cell->at_boundary())
	   {
		  for (unsigned int face_num=0; face_num < GeometryInfo<dim>::faces_per_cell; ++face_num)
		  {
			 if ( cell->face(face_num)->at_boundary() )
			 {
			    hp_fe_face_v.reinit(cell,face_num);
			    const FEFaceValues<dim> &fe_face_v = hp_fe_face_v.get_present_fe_values();
			    const Quadrature<dim-1> &face_quadrature = hp_fe_face_v.get_present_fe_values().get_quadrature();
			    std::vector<Point<dim>> f_q_points(face_quadrature.size());
			    f_q_points = fe_face_v.get_quadrature_points();

			    Assert (dofs_per_cell == dofs_per_cell_0, ExcInternalError());

			    std::vector<Tensor<1,dim>> tensors_disp_fluctuations(face_quadrature.size()), tensors_lambda(face_quadrature.size());
			    fe_face_v[displacements].get_function_values(temp_solution,tensors_disp_fluctuations);
			    fe_face_v[multipliers].get_function_values(temp_solution,tensors_lambda);

			    for (unsigned int f_qp=0; f_qp<f_q_points.size(); ++f_qp)
			    {
			       for (unsigned int i=0; i<dofs_per_cell; ++i)
			       {
			    	  Tensor<1,dim> n_i_u = fe_face_v[displacements].value(i,f_qp);
			    	  Tensor<1,dim> n_i_lambda = fe_face_v[multipliers].value(i,f_qp);

/*			    	  if (cell->face(face_num)->boundary_id()==0)
			    	  {
			    		 cell_rhs[i] += reciprocal_vol * (n_i_u * tensors_lambda[f_qp]) * fe_face_v.JxW(f_qp);
			    	     cell_rhs[i] += reciprocal_vol * (n_i_lambda * tensors_disp_fluctuations[f_qp]) * fe_face_v.JxW(f_qp);
			    	  }
			    	  else
			    	  {
			    		 cell_rhs[i] -= reciprocal_vol * (n_i_u * tensors_lambda[f_qp]) * fe_face_v.JxW(f_qp);
			    		 cell_rhs[i] -= reciprocal_vol * (n_i_lambda * tensors_disp_fluctuations[f_qp]) * fe_face_v.JxW(f_qp);
			    	  }*/
			    	  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    	  {
			    		 Tensor<1,dim> n_j_u = fe_face_v[displacements].value(j,f_qp);
			    		 Tensor<1,dim> n_j_lambda = fe_face_v[multipliers].value(j,f_qp);

				    	  if (cell->face(face_num)->boundary_id()==0)
				    	  {
					    	 cell_matrix(i,j) += reciprocal_vol * (n_i_u * n_j_lambda) * fe_face_v.JxW(f_qp);
					    	 cell_matrix(i,j) += reciprocal_vol * (n_i_lambda * n_j_u) * fe_face_v.JxW(f_qp);
				    	  }
				    	  else
				    	  {
						     cell_matrix(i,j) -= reciprocal_vol * (n_i_u * n_j_lambda) * fe_face_v.JxW(f_qp);
						     cell_matrix(i,j) -= reciprocal_vol * (n_i_lambda * n_j_u) * fe_face_v.JxW(f_qp);
				    	  }
			    	  }
			       }
			    }
			 }
		  }
	   }
	   for (unsigned int i=0; i<dofs_per_cell; ++i)
	   {
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
			  if (local_dof_indices[i] < matrix.m() && local_dof_indices[j] < matrix.n())
				 matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
		  }
		  if (local_dof_indices[i] < rhs.size())
		    rhs[local_dof_indices[i]] += cell_rhs[i];
	   }
	}
  }



  template <int dim>
  void ElasticProblem<dim>::assemble_system_pDBC_2(BlockSparseMatrix<double> &matrix,
  						  	  	  	  	  	  	   const SymmetricTensor<2,dim> &epsilon,
												   BlockVector<double> &rhs) const
  {
	 BlockVector<double> temp_solution(dofs_per_block);
     matrix = 0;
	 rhs = 0;

	 const double reciprocal_vol = 1 / GridTools::volume(triangulation);

	 hp::FEValues<dim> hp_fe_v(fe_collection,quadrature_collection,
			 	 	 	 	   update_values | update_quadrature_points | update_JxW_values | update_gradients);
	 hp::FEFaceValues<dim> hp_fe_face_v(fe_collection,face_quadrature_collection,
			 	 	 	 	 	 	 	update_values | update_quadrature_points | update_JxW_values | update_gradients);

 	 std::vector<dealii::types::global_dof_index> local_dof_indices;

	 unsigned int dofs_per_cell_0 = fe_collection[0].dofs_per_cell;
	 unsigned int dofs_per_cell_1 = fe_collection[1].dofs_per_cell;

	 FullMatrix<double> cell_matrix, cell_matrix_2(dofs_per_cell_0,dim);
	 Vector<double> cell_rhs;

	 typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	 typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	 for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
	 {
		 const unsigned int active_index = cell->active_fe_index();

	   hp_fe_v.reinit (cell);
	   const FEValues<dim> &fe_v = hp_fe_v.get_present_fe_values();

	   unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	   local_dof_indices.resize(dofs_per_cell);
	   cell->get_dof_indices(local_dof_indices);
	   cell_matrix.reinit(local_dof_indices.size(),local_dof_indices.size());
	   cell_rhs.reinit(local_dof_indices.size());

	   cell_matrix = 0;
	   cell_rhs = 0;
	   cell_matrix_2 = 0;

	   std::vector<SymmetricTensor<2,dim>> tensor_eps(fe_v.n_quadrature_points), tensors_sigma(fe_v.n_quadrature_points);
	   fe_v[displacements].get_function_symmetric_gradients(temp_solution,tensor_eps);
	   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
		   tensor_eps[q_p] += epsilon;

	   unsigned int material_id = cell->material_id();
	   SymmetricTensor<4,dim> c_4;
	   if (material_id==2)
	     c_4 = c4_matrix;
	   else if (material_id==1)
		 c_4 = c4_inclusion;
	   else
		 AssertThrow(false,ExcMessage("invalid material id"));

	   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
	   {
		  tensors_sigma[q_p] = c_4 * tensor_eps[q_p];

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
		     SymmetricTensor<2,dim> sym_grad_N_i = fe_v[displacements].symmetric_gradient(i,q_p);

		     cell_rhs[i] -= sym_grad_N_i * tensors_sigma[q_p] * fe_v.JxW(q_p);

		     for (unsigned int j=0; j<dofs_per_cell; ++j)
		     {
		    	SymmetricTensor<2,dim> sym_grad_N_j = fe_v[displacements].symmetric_gradient(j,q_p);
		    	SymmetricTensor<2,dim> sigma_j = c_4 * sym_grad_N_j;

		    	cell_matrix(i,j) += (sym_grad_N_i * sigma_j) * fe_v.JxW(q_p);
		     }
		  }


		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
			  std::pair<unsigned int,types::global_dof_index> system_to_block = fe_collection[active_index].system_to_block_index(i);
			  Tensor<1,dim> disp_i = fe_v[displacements].value(i,q_p);

			  for (unsigned int j=0; j<dim; ++j)
			  {
				  double entry = disp_i[j] * fe_v.JxW(q_p);

				  cell_matrix_2(system_to_block.second,j) += entry;
			  }
		  }
	   }
	   if (cell->at_boundary())
	   {
		  for (unsigned int face_num=0; face_num < GeometryInfo<dim>::faces_per_cell; ++face_num)
		  {
			 if ( cell->face(face_num)->at_boundary() )
			 {
			    hp_fe_face_v.reinit(cell,face_num);
			    const FEFaceValues<dim> &fe_face_v = hp_fe_face_v.get_present_fe_values();
			    const Quadrature<dim-1> &face_quadrature = hp_fe_face_v.get_present_fe_values().get_quadrature();
			    std::vector<Point<dim>> f_q_points(face_quadrature.size());
			    f_q_points = fe_face_v.get_quadrature_points();

			    Assert (dofs_per_cell == dofs_per_cell_0, ExcInternalError());

			    std::vector<Tensor<1,dim>> tensors_disp_fluctuations(face_quadrature.size()), tensors_lambda(face_quadrature.size());
			    fe_face_v[displacements].get_function_values(temp_solution,tensors_disp_fluctuations);
			    fe_face_v[multipliers].get_function_values(temp_solution,tensors_lambda);

			    for (unsigned int f_qp=0; f_qp<f_q_points.size(); ++f_qp)
			    {
			       for (unsigned int i=0; i<dofs_per_cell; ++i)
			       {
			    	  Tensor<1,dim> n_i_u = fe_face_v[displacements].value(i,f_qp);
			    	  Tensor<1,dim> n_i_lambda = fe_face_v[multipliers].value(i,f_qp);

/*			    	  if (cell->face(face_num)->boundary_id()==0)
			    	  {
			    		 cell_rhs[i] += reciprocal_vol * (n_i_u * tensors_lambda[f_qp]) * fe_face_v.JxW(f_qp);
			    	     cell_rhs[i] += reciprocal_vol * (n_i_lambda * tensors_disp_fluctuations[f_qp]) * fe_face_v.JxW(f_qp);
			    	  }
			    	  else
			    	  {
			    		 cell_rhs[i] -= reciprocal_vol * (n_i_u * tensors_lambda[f_qp]) * fe_face_v.JxW(f_qp);
			    		 cell_rhs[i] -= reciprocal_vol * (n_i_lambda * tensors_disp_fluctuations[f_qp]) * fe_face_v.JxW(f_qp);
			    	  }*/
			    	  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    	  {
			    		 Tensor<1,dim> n_j_u = fe_face_v[displacements].value(j,f_qp);
			    		 Tensor<1,dim> n_j_lambda = fe_face_v[multipliers].value(j,f_qp);

				    	  if (cell->face(face_num)->boundary_id()==0)
				    	  {
					    	 cell_matrix(i,j) += reciprocal_vol * (n_i_u * n_j_lambda) * fe_face_v.JxW(f_qp);
					    	 cell_matrix(i,j) += reciprocal_vol * (n_i_lambda * n_j_u) * fe_face_v.JxW(f_qp);
				    	  }
				    	  else
				    	  {
						     cell_matrix(i,j) -= reciprocal_vol * (n_i_u * n_j_lambda) * fe_face_v.JxW(f_qp);
						     cell_matrix(i,j) -= reciprocal_vol * (n_i_lambda * n_j_u) * fe_face_v.JxW(f_qp);
				    	  }
			    	  }
			       }
			    }
			 }
		  }
	   }
	   for (unsigned int i=0; i<dofs_per_cell; ++i)
	   {
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
			  if (local_dof_indices[i] < matrix.m() && local_dof_indices[j] < matrix.n())
				 matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
		  }
		  if (local_dof_indices[i] < rhs.size())
		    rhs[local_dof_indices[i]] += cell_rhs[i];
	   }
	   for (unsigned int i=0; i<dofs_per_cell; ++i)
	   {
		   std::pair<unsigned int,types::global_dof_index> system_to_block = fe_collection[active_index].system_to_block_index(i);

		   if (system_to_block.first==0)
			   for (unsigned int j=0; j<dim; ++j)
			   {
				   const unsigned int global_index_i = local_dof_indices[i];
				   const unsigned int global_index_j = matrix.block(0,0).n() + matrix.block(0,1).n() + j;

				   matrix.add(global_index_i,global_index_j,cell_matrix_2(system_to_block.second,j));

				   matrix.add(global_index_j,global_index_i,cell_matrix_2(system_to_block.second,j));
			   }
	   }
	}
  }


  template <int dim>
  void ElasticProblem<dim>::solve(BlockSparseMatrix<double> &matrix,
		   	   	   	   	   	   	  const BlockVector<double> &rhs,
								  BlockVector<double> &solution) const
  {
	 SparseDirectUMFPACK direct_solver;
	 direct_solver.initialize(matrix);
	 direct_solver.vmult(solution,rhs);
  }


  template <int dim>
  void ElasticProblem<dim>::output(const BlockVector<double> &solution,
		  	  	  	  	  	  	   const std::string &file_name,
								   const SymmetricTensor<2,dim> &epsilon) const
  {
	 BlockVector<double> makro_solution(solution);
	 MakroField<dim> makro_field(epsilon);
	 //VectorTools::interpolate(dof_handler,makro_field,makro_solution);
	 makro_solution += solution;

	 Vector<double> cell_material_ids(triangulation.n_active_cells());
	 typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	 typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	 for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
		 cell_material_ids[cell_num] = cell->material_id();

     std::ofstream output (file_name.c_str());
     std::vector<std::string> solution_names (dim, "displacement");
     for (unsigned int i=0; i<dim; ++i)
       solution_names.push_back ("lagrange_multiplier");
	 std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(2*dim,DataComponentInterpretation::component_is_part_of_vector);
	 DataOut<dim,hp::DoFHandler<dim>> data_out;
	 data_out.attach_dof_handler(dof_handler);
	 data_out.add_data_vector(makro_solution, solution_names,
							  DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
		                      data_component_interpretation);
	 data_out.add_data_vector(cell_material_ids,"material_id");
	 data_out.build_patches();
	 data_out.write_vtu(output/*, DataOutBase::OutputFormat::vtk*/);
  }


  template <int dim>
  SymmetricTensor<2,dim> ElasticProblem<dim>::averaged_sigma(const BlockVector<double> &solution,
  									  	  	  	  	  	  	 const SymmetricTensor<2,dim> &epsilon)
  {
	  SymmetricTensor<2,dim> sigma;
	  const double reciprocal_vol = 1 / GridTools::volume(triangulation);

	  hp::FEValues<dim> hp_fe_v(fe_collection,quadrature_collection,
			  	  	  	  	  	update_values | update_quadrature_points | update_JxW_values | update_gradients);

	  typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	  typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	  for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
	  {
		   hp_fe_v.reinit (cell);
		   const FEValues<dim> &fe_v = hp_fe_v.get_present_fe_values();

		   std::vector<SymmetricTensor<2,dim>> tensor_eps(fe_v.n_quadrature_points), tensors_sigma(fe_v.n_quadrature_points);
		   fe_v[displacements].get_function_symmetric_gradients(solution,tensor_eps);

		   unsigned int material_id = cell->material_id();
		   SymmetricTensor<4,dim> c_4;
		   if (material_id==2)
		     c_4 = c4_matrix;
		   else if (material_id==1)
			 c_4 = c4_inclusion;
		   else
			 AssertThrow(false,ExcMessage("invalid material id"));

		   for (unsigned int q_p=0; q_p<fe_v.n_quadrature_points; ++q_p)
		   {
			  tensor_eps[q_p] += epsilon;
			  tensors_sigma[q_p] = c_4 * tensor_eps[q_p];

			  sigma += tensors_sigma[q_p] * fe_v.JxW(q_p);
		   }
		}
	  sigma *= reciprocal_vol;
	  return sigma;
  }


  template <int dim>
  void ElasticProblem<dim>::refine()
  {
/*	 Vector<double> interface_cells(triangulation.n_active_cells());

	 typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	 typename hp::DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	 for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
	 {
	    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	       if (!cell->face(face)->at_boundary())
	       {
	    	   typename hp::DoFHandler<dim>::active_cell_iterator neighbor = cell->neighbor(face);

	    	   if (cell->material_id() != neighbor->material_id())
	    	   {
	    		   interface_cells[cell_num] = 2;

	    		   cell->set_all_manifold_ids(17);
	    	   }
	       }
	    }
	 }
	 const types::manifold_id manifold_number = 17;
	 Point<dim> center;
	 center[0] = 0.5;
	 center[1] = 0.5;
	 const SphericalManifold<2> interface_description(center);
	 triangulation.set_manifold(manifold_number,interface_description);

	 {
		Vector<double> cell_material_ids(triangulation.n_active_cells());
		cell = dof_handler.begin_active();
		endc = dof_handler.end();
		for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
		  cell_material_ids[cell_num] = cell->material_id();

	    std::string file_name = "interfaces_0.vtu" ;
	    std::ofstream output(file_name.c_str());

		DataOut<dim,hp::DoFHandler<dim>> data_out;
		data_out.attach_dof_handler(dof_handler);

		data_out.add_data_vector(cell_material_ids,"material_id");
		data_out.add_data_vector(interface_cells,"interface_cells");
		data_out.build_patches();
		data_out.write_vtu(output, DataOutBase::OutputFormat::vtk);
	 }
	 triangulation.refine_global(4);
	 {
		Vector<double> cell_material_ids(triangulation.n_active_cells());
		cell = dof_handler.begin_active();
		endc = dof_handler.end();
		for (unsigned int cell_num=0; cell != endc; ++cell, ++cell_num)
		  cell_material_ids[cell_num] = cell->material_id();

	    std::string file_name = "interfaces_1.vtu" ;
	    std::ofstream output(file_name.c_str());

		DataOut<dim,hp::DoFHandler<dim>> data_out;
		data_out.attach_dof_handler(dof_handler);

		data_out.add_data_vector(cell_material_ids,"material_id");
		data_out.build_patches();
		data_out.write_vtu(output, DataOutBase::OutputFormat::vtk);
	 }
	 triangulation.set_manifold(manifold_number);*/
  }

  template <int dim>
  void ElasticProblem<dim>::run_I(const SymmetricTensor<2,dim> &epsilon)
  {
	  setup_system_1();

	  BlockSparseMatrix<double> matrix_lDBC(sparsity_pattern);
	  BlockVector<double> tmp_solution(relevant_dofs_per_block), rhs(relevant_dofs_per_block);

	  assemble_system_lDBC(matrix_lDBC,epsilon,rhs);

	  solve(matrix_lDBC,rhs,tmp_solution);

	  BlockVector<double> solution(dofs_per_block);
	  for (unsigned int i=0; i<tmp_solution.size(); ++i)
	    solution[i] = tmp_solution[i];

	  SymmetricTensor<2,dim> av_sigma = averaged_sigma(solution,epsilon);
	  std::cout << "lDBC: volume average of sigma: " << av_sigma << std::endl;

	  output(solution,"solution_lDBC.vtu",epsilon);
  }


  template <int dim>
  void ElasticProblem<dim>::run_II(const SymmetricTensor<2,dim> &epsilon)
  {
	  setup_system_1();

	  BlockSparseMatrix<double> matrix_pDBC(sparsity_pattern);
	  Vector<double> tmp_solution(relevant_dofs_per_block[0]+relevant_dofs_per_block[1]),
			  	  	 tmp_rhs(relevant_dofs_per_block[0]+relevant_dofs_per_block[1]);
	  BlockVector<double> rhs(relevant_dofs_per_block);

	  assemble_system_pDBC_1(matrix_pDBC,epsilon,rhs);
	  tmp_rhs = rhs;

	  LAPACKFullMatrix<double> dense_matrix_pDBC(matrix_pDBC.m(),matrix_pDBC.n());

	  for (BlockSparseMatrix<double>::const_iterator it = matrix_pDBC.begin(); it != matrix_pDBC.end(); ++it)
		  dense_matrix_pDBC(it->row(),it->column()) = it->value();

	  dense_matrix_pDBC.compute_svd();

	  const double ratio = 1e-8;
	  unsigned int rank_pDBC=1;
	  for (unsigned int i=1; i<std::min(dense_matrix_pDBC.m(),dense_matrix_pDBC.n()); ++i)
		  if (dense_matrix_pDBC.singular_value(i) >=  dense_matrix_pDBC.singular_value(0)*ratio)
			  ++rank_pDBC;
		  else
			  break;

	  std::cout << std::endl << std::endl;
	  std::cout << "numerical rank of matrix: " << rank_pDBC << "/" << std::min(dense_matrix_pDBC.m(),dense_matrix_pDBC.n()) << std::endl;

	  dense_matrix_pDBC.compute_inverse_svd(ratio);
	  dense_matrix_pDBC.vmult(tmp_solution,tmp_rhs);

	  BlockVector<double> solution(dofs_per_block);
	  for (unsigned int i=0; i<tmp_solution.size(); ++i)
	    solution[i] = tmp_solution[i];

	  SymmetricTensor<2,dim> av_sigma = averaged_sigma(solution,epsilon);
	  std::cout << "pDBC I: volume average of sigma: " << av_sigma << std::endl;

	  output(solution,"solution_pDBC_1.vtu",epsilon);
  }



  template <int dim>
  void ElasticProblem<dim>::run_III(const SymmetricTensor<2,dim> &epsilon)
  {
	  setup_system_2();

	  BlockSparseMatrix<double> matrix_pDBC(sparsity_pattern);
	  Vector<double> tmp_solution(relevant_dofs_per_block[0]+relevant_dofs_per_block[1]),
			  	  	 tmp_rhs(relevant_dofs_per_block[0]+relevant_dofs_per_block[1]);
	  BlockVector<double> rhs(relevant_dofs_per_block),solution(relevant_dofs_per_block);

	  assemble_system_pDBC_2(matrix_pDBC,epsilon,rhs);
	  solve(matrix_pDBC,rhs,solution);

	  BlockVector<double> fe_solution(dofs_per_block);
	  for (unsigned int i=0; i<tmp_solution.size(); ++i)
		  fe_solution[i] = solution[i];

	  output(fe_solution,"solution_pDBC_2.vtu",epsilon);

	  SymmetricTensor<2,dim> av_sigma = averaged_sigma(fe_solution,epsilon);
	  std::cout << "pDBC 2: volume average of sigma: " << av_sigma << std::endl;

/*	  LAPACKFullMatrix<double> dense_matrix_pDBC(matrix_pDBC.m(),matrix_pDBC.n());

	  for (BlockSparseMatrix<double>::const_iterator it = matrix_pDBC.begin(); it != matrix_pDBC.end(); ++it)
		  dense_matrix_pDBC(it->row(),it->column()) = it->value();

	  dense_matrix_pDBC.compute_svd();

	  const double ratio = 1e-8;
	  unsigned int rank_pDBC=1;
	  for (unsigned int i=1; i<std::min(dense_matrix_pDBC.m(),dense_matrix_pDBC.n()); ++i)
		  if (dense_matrix_pDBC.singular_value(i) >=  dense_matrix_pDBC.singular_value(0)*ratio)
			  ++rank_pDBC;
		  else
			  break;

	  std::cout << std::endl << std::endl;
	  std::cout << "numerical rank of matrix: " << rank_pDBC << "/" << std::min(dense_matrix_pDBC.m(),dense_matrix_pDBC.n()) << std::endl;

	  dense_matrix_pDBC.compute_inverse_svd(ratio);
	  dense_matrix_pDBC.vmult(tmp_solution,tmp_rhs);

	  BlockVector<double> solution(dofs_per_block);
	  for (unsigned int i=0; i<tmp_solution.size(); ++i)
	    solution[i] = tmp_solution[i];

	  SymmetricTensor<2,dim> av_sigma = averaged_sigma(solution,epsilon);
	  std::cout << "DoFs per block: " << dofs_per_block[0] << " & " << dofs_per_block[1] << std::endl;
	  std::cout << "relevant DoFs per block: " << relevant_dofs_per_block[0] << " & " << relevant_dofs_per_block[1] << std::endl;
	  std::cout << "pDBC I: volume average of sigma: " << av_sigma << std::endl;

	  output(solution,"solution_pDBC_1.vtu");*/
  }



}



int main ()
{
  try
    {
	  dealii::SymmetricTensor<2,2> epsilon;
	  epsilon[0][0] = 0.1;
	  epsilon[0][1] = 0.2;
	  epsilon[1][1] = 0.05;

      Elasticity::ElasticProblem<2> elastic_problem_2d_I(2,"/home/benjamin/PA_Feng/mesh.inp");
      elastic_problem_2d_I.run_I(epsilon);

      //Elasticity::ElasticProblem<2> elastic_problem_2d_II(2,"/home/benjamin/PA_Feng/mesh.inp");
      //elastic_problem_2d_II.run_II(epsilon);

      Elasticity::ElasticProblem<2> elastic_problem_2d_III(2,"/home/benjamin/PA_Feng/mesh.inp");
      elastic_problem_2d_III.run_III(epsilon);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
