<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>compute.hpp</name>
    <path>irlba/</path>
    <filename>compute_8hpp.html</filename>
    <includes id="simple_8hpp" name="simple.hpp" local="yes" import="no" module="no" objc="no">Matrix/simple.hpp</includes>
    <class kind="struct">irlba::Results</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>irlba.hpp</name>
    <path>irlba/</path>
    <filename>irlba_8hpp.html</filename>
    <includes id="compute_8hpp" name="compute.hpp" local="yes" import="no" module="no" objc="no">compute.hpp</includes>
    <includes id="pca_8hpp" name="pca.hpp" local="yes" import="no" module="no" objc="no">pca.hpp</includes>
    <includes id="parallel_8hpp" name="parallel.hpp" local="yes" import="no" module="no" objc="no">parallel.hpp</includes>
    <includes id="Options_8hpp" name="Options.hpp" local="yes" import="no" module="no" objc="no">Options.hpp</includes>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>centered.hpp</name>
    <path>irlba/Matrix/</path>
    <filename>centered_8hpp.html</filename>
    <includes id="interface_8hpp" name="interface.hpp" local="yes" import="no" module="no" objc="no">interface.hpp</includes>
    <class kind="class">irlba::CenteredWorkspace</class>
    <class kind="class">irlba::CenteredAdjointWorkspace</class>
    <class kind="class">irlba::CenteredRealizeWorkspace</class>
    <class kind="class">irlba::CenteredMatrix</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>interface.hpp</name>
    <path>irlba/Matrix/</path>
    <filename>interface_8hpp.html</filename>
    <class kind="class">irlba::Workspace</class>
    <class kind="class">irlba::AdjointWorkspace</class>
    <class kind="class">irlba::RealizeWorkspace</class>
    <class kind="class">irlba::Matrix</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>scaled.hpp</name>
    <path>irlba/Matrix/</path>
    <filename>scaled_8hpp.html</filename>
    <includes id="interface_8hpp" name="interface.hpp" local="yes" import="no" module="no" objc="no">interface.hpp</includes>
    <class kind="class">irlba::ScaledWorkspace</class>
    <class kind="class">irlba::ScaledAdjointWorkspace</class>
    <class kind="class">irlba::ScaledRealizeWorkspace</class>
    <class kind="class">irlba::ScaledMatrix</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>simple.hpp</name>
    <path>irlba/Matrix/</path>
    <filename>simple_8hpp.html</filename>
    <includes id="interface_8hpp" name="interface.hpp" local="yes" import="no" module="no" objc="no">interface.hpp</includes>
    <class kind="class">irlba::SimpleWorkspace</class>
    <class kind="class">irlba::SimpleAdjointWorkspace</class>
    <class kind="class">irlba::SimpleRealizeWorkspace</class>
    <class kind="class">irlba::SimpleMatrix</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>sparse.hpp</name>
    <path>irlba/Matrix/</path>
    <filename>sparse_8hpp.html</filename>
    <includes id="parallel_8hpp" name="parallel.hpp" local="yes" import="no" module="no" objc="no">../parallel.hpp</includes>
    <includes id="interface_8hpp" name="interface.hpp" local="yes" import="no" module="no" objc="no">interface.hpp</includes>
    <class kind="class">irlba::ParallelSparseWorkspace</class>
    <class kind="class">irlba::ParallelSparseAdjointWorkspace</class>
    <class kind="class">irlba::ParallelSparseRealizeWorkspace</class>
    <class kind="class">irlba::ParallelSparseMatrix</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>Options.hpp</name>
    <path>irlba/</path>
    <filename>Options_8hpp.html</filename>
    <class kind="struct">irlba::Options</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>parallel.hpp</name>
    <path>irlba/</path>
    <filename>parallel_8hpp.html</filename>
    <class kind="class">irlba::EigenThreadScope</class>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="file">
    <name>pca.hpp</name>
    <path>irlba/</path>
    <filename>pca_8hpp.html</filename>
    <includes id="compute_8hpp" name="compute.hpp" local="yes" import="no" module="no" objc="no">compute.hpp</includes>
    <includes id="simple_8hpp" name="simple.hpp" local="yes" import="no" module="no" objc="no">Matrix/simple.hpp</includes>
    <includes id="centered_8hpp" name="centered.hpp" local="yes" import="no" module="no" objc="no">Matrix/centered.hpp</includes>
    <includes id="scaled_8hpp" name="scaled.hpp" local="yes" import="no" module="no" objc="no">Matrix/scaled.hpp</includes>
    <namespace>irlba</namespace>
  </compound>
  <compound kind="class">
    <name>irlba::AdjointWorkspace</name>
    <filename>classirlba_1_1AdjointWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1AdjointWorkspace.html</anchorfile>
      <anchor>a586c1f21819b781503a58c87249c3b63</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)=0</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::CenteredAdjointWorkspace</name>
    <filename>classirlba_1_1CenteredAdjointWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Center_</templarg>
    <base>irlba::AdjointWorkspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1CenteredAdjointWorkspace.html</anchorfile>
      <anchor>a62336e2c0fb751243a638f9bf3c5c8c5</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;out)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::CenteredMatrix</name>
    <filename>classirlba_1_1CenteredMatrix.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class MatrixPointer_</templarg>
    <templarg>class CenterPointer_</templarg>
    <base>irlba::Matrix&lt; EigenVector_, EigenMatrix_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>CenteredMatrix</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>ae2d0db17316cb8865003c01177cd94f0</anchor>
      <arglist>(const MatrixPointer_ &amp;matrix, const CenterPointer_ &amp;center)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>rows</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>a7bd996660d56400904d1191d3d9dee96</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>cols</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>a2ccca89f9ed6d8a609503b0833cc945b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>a7dbbce4a13a2ef4051fe223a8cb8183f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_adjoint_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>acd389d1385dacd65330aa3b8c6f7718e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_realize_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>abc897e918a23c24eeaedfe43760d151f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; CenteredWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_center)&gt; &gt; &gt;</type>
      <name>new_known_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>a2c7fa29c1575e39c2f57c83ef4ac442d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; CenteredAdjointWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_center)&gt; &gt; &gt;</type>
      <name>new_known_adjoint_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>aaf02c11ab9f33e80d2c2ca7a7e4aab5d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; CenteredRealizeWorkspace&lt; EigenMatrix_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_center)&gt; &gt; &gt;</type>
      <name>new_known_realize_workspace</name>
      <anchorfile>classirlba_1_1CenteredMatrix.html</anchorfile>
      <anchor>a899c39be9fcaa885177610056ff3411c</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::CenteredRealizeWorkspace</name>
    <filename>classirlba_1_1CenteredRealizeWorkspace.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Center_</templarg>
    <base>irlba::RealizeWorkspace&lt; EigenMatrix_ &gt;</base>
    <member kind="function">
      <type>const EigenMatrix_ &amp;</type>
      <name>realize</name>
      <anchorfile>classirlba_1_1CenteredRealizeWorkspace.html</anchorfile>
      <anchor>a6aece4f93699ca7d925cf7e62ff492f1</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::CenteredWorkspace</name>
    <filename>classirlba_1_1CenteredWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Center_</templarg>
    <base>irlba::Workspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1CenteredWorkspace.html</anchorfile>
      <anchor>a7f7bddfea87ce7660c27ce57f319646c</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;out)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::EigenThreadScope</name>
    <filename>classirlba_1_1EigenThreadScope.html</filename>
    <member kind="function">
      <type></type>
      <name>EigenThreadScope</name>
      <anchorfile>classirlba_1_1EigenThreadScope.html</anchorfile>
      <anchor>a5bed6285b3da4aba87eeaf3cddbef1c1</anchor>
      <arglist>(int num_threads)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::Matrix</name>
    <filename>classirlba_1_1Matrix.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class EigenMatrix_</templarg>
    <member kind="function" virtualness="pure">
      <type>virtual Eigen::Index</type>
      <name>rows</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a445f7873a357819428df497c3ed69a33</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual Eigen::Index</type>
      <name>cols</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a501fe6e1d2f916d46239529e731da328</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a6abf35c662b5f18dddfa1a5e68a3d227</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_adjoint_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a21e0ea4abfe872d6c0d6730a997db828</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function" virtualness="pure">
      <type>virtual std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_realize_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a34bc2683917efaf71525446eef8b7786</anchor>
      <arglist>() const =0</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_known_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>af4069e1f696ae7495d1b495d08fd57a1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_known_adjoint_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a4d24b11b4a1f7082f88fae5dfc70edd4</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_known_realize_workspace</name>
      <anchorfile>classirlba_1_1Matrix.html</anchorfile>
      <anchor>a9a4da990893ff44f8b2039d2acfa4332</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>irlba::Options</name>
    <filename>structirlba_1_1Options.html</filename>
    <member kind="variable">
      <type>double</type>
      <name>invariant_subspace_tolerance</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>ad746aa1f21cf2b8b5eb50500ccf1d4bc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>convergence_tolerance</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>a1fd00b6a91fb447f35c0c0cfa65dafd7</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>double</type>
      <name>singular_value_ratio_tolerance</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>a1e6e7fa2394b489f39e613c0f0fa3092</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>extra_work</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>ab903b879a43cb167ddbab093b44c32e0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>max_iterations</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>afc7f9a04a63cc7033c2103cdebab3612</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>exact_for_small_matrix</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>a6139d421c988fe1818bfe338f9ce85ad</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>exact_for_large_number</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>ae4bd4e00dd4815e3aa66b0b23f8820b0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>cap_number</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>a4e6e040a6ff921f48c70961ad9876c31</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>uint64_t</type>
      <name>seed</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>a4252ea1bbebe8ad2a7541c1edb2699e2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>void *</type>
      <name>initial</name>
      <anchorfile>structirlba_1_1Options.html</anchorfile>
      <anchor>ab7d48ae8392f7097519642f5b2ffe26c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ParallelSparseAdjointWorkspace</name>
    <filename>classirlba_1_1ParallelSparseAdjointWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class ValueArray_</templarg>
    <templarg>class IndexArray_</templarg>
    <templarg>class PointerArray_</templarg>
    <base>irlba::AdjointWorkspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1ParallelSparseAdjointWorkspace.html</anchorfile>
      <anchor>a6e8b220c427cbbc75d2d3bc00f804d88</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ParallelSparseMatrix</name>
    <filename>classirlba_1_1ParallelSparseMatrix.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class ValueArray_</templarg>
    <templarg>class IndexArray_</templarg>
    <templarg>class PointerArray_</templarg>
    <base>irlba::Matrix&lt; EigenVector_, EigenMatrix_ &gt;</base>
    <member kind="typedef">
      <type>I&lt; decltype(std::declval&lt; PointerArray_ &gt;()[0])&gt;</type>
      <name>PointerType</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a812d625d449ea75df1b42428a0d09c03</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ParallelSparseMatrix</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a95f1f0793968638d2a7086beb2b6dab2</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>ParallelSparseMatrix</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a396d19eeb0b384636072b6fe41512ad0</anchor>
      <arglist>(Eigen::Index nrow, Eigen::Index ncol, ValueArray_ x, IndexArray_ i, PointerArray_ p, bool column_major, int num_threads)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>rows</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a7ea7b264d343553c52cb4e0a04bf0cf9</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>cols</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a4384154be9eb26ee6d789e749c33390f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const ValueArray_ &amp;</type>
      <name>get_values</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a78d57d5921d40f01a45543bb6b9cd277</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const IndexArray_ &amp;</type>
      <name>get_indices</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>adaf070a354be6f2d65263864cc3cea15</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const PointerArray_ &amp;</type>
      <name>get_pointers</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>ad28e10195596f59f6f089c1e89f3f2ea</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; size_t &gt; &amp;</type>
      <name>get_primary_starts</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a7b0edebfb5df14167ce08100d19e824f</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; size_t &gt; &amp;</type>
      <name>get_primary_ends</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>aa6f051878910af5820563453cb8e0957</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; std::vector&lt; PointerType &gt; &gt; &amp;</type>
      <name>get_secondary_nonzero_starts</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a5b338eba3c51b12314bcaeb506dee6f1</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>adf44a97e086bb8446961ff36faa9f509</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_adjoint_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a7d9b58d608a2b2b8a8b8cbb4b9e41ce8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_realize_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>a9d7a53e223d10e5e5863a24f3ab82b4e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ParallelSparseWorkspace&lt; EigenVector_, ValueArray_, IndexArray_, PointerArray_ &gt; &gt;</type>
      <name>new_known_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>ac3653e6ecd6ef7c8b2455880279e705c</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ParallelSparseAdjointWorkspace&lt; EigenVector_, ValueArray_, IndexArray_, PointerArray_ &gt; &gt;</type>
      <name>new_known_adjoint_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>afbd674f5821082d248fab3ed0357d8c0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ParallelSparseRealizeWorkspace&lt; EigenMatrix_, ValueArray_, IndexArray_, PointerArray_ &gt; &gt;</type>
      <name>new_known_realize_workspace</name>
      <anchorfile>classirlba_1_1ParallelSparseMatrix.html</anchorfile>
      <anchor>ad55f7e2ad547885ab8dcc78fd819b795</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ParallelSparseRealizeWorkspace</name>
    <filename>classirlba_1_1ParallelSparseRealizeWorkspace.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class ValueArray_</templarg>
    <templarg>class IndexArray_</templarg>
    <templarg>class PointerArray_</templarg>
    <base>irlba::RealizeWorkspace&lt; EigenMatrix_ &gt;</base>
    <member kind="function">
      <type>const EigenMatrix_ &amp;</type>
      <name>realize</name>
      <anchorfile>classirlba_1_1ParallelSparseRealizeWorkspace.html</anchorfile>
      <anchor>a7f1d0da3da9d067a53cd08006a7affb6</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ParallelSparseWorkspace</name>
    <filename>classirlba_1_1ParallelSparseWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class ValueArray_</templarg>
    <templarg>class IndexArray_</templarg>
    <templarg>class PointerArray_</templarg>
    <base>irlba::Workspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1ParallelSparseWorkspace.html</anchorfile>
      <anchor>ad3fc5f626b0bb46c158b5815be9d8482</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::RealizeWorkspace</name>
    <filename>classirlba_1_1RealizeWorkspace.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <member kind="function" virtualness="pure">
      <type>virtual const EigenMatrix_ &amp;</type>
      <name>realize</name>
      <anchorfile>classirlba_1_1RealizeWorkspace.html</anchorfile>
      <anchor>a8596db8855716e629a8510ac6e12e71e</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)=0</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>realize_copy</name>
      <anchorfile>classirlba_1_1RealizeWorkspace.html</anchorfile>
      <anchor>afbd1cdc749cdca173d5659bb9a8718c1</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>irlba::Results</name>
    <filename>structirlba_1_1Results.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class EigenVector_</templarg>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>U</name>
      <anchorfile>structirlba_1_1Results.html</anchorfile>
      <anchor>a59ab34dac65eb90e6886b8a7df6ec245</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>V</name>
      <anchorfile>structirlba_1_1Results.html</anchorfile>
      <anchor>a6559892535ea2ba3b80bfd101e979f3b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>D</name>
      <anchorfile>structirlba_1_1Results.html</anchorfile>
      <anchor>a21b7e432392523d3e6693701751a4cfc</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>iterations</name>
      <anchorfile>structirlba_1_1Results.html</anchorfile>
      <anchor>a5788fc587a38a96d46bd75bc6c180559</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>converged</name>
      <anchorfile>structirlba_1_1Results.html</anchorfile>
      <anchor>a867a16b6a3c46b48deea9a17dd4e20b9</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ScaledAdjointWorkspace</name>
    <filename>classirlba_1_1ScaledAdjointWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Scale_</templarg>
    <base>irlba::AdjointWorkspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1ScaledAdjointWorkspace.html</anchorfile>
      <anchor>aa9d658babc15da2738e7303e54f8a00c</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;out)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ScaledMatrix</name>
    <filename>classirlba_1_1ScaledMatrix.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class MatrixPointer_</templarg>
    <templarg>class ScalePointer_</templarg>
    <base>irlba::Matrix&lt; EigenVector_, EigenMatrix_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>ScaledMatrix</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>a48c27c2c1b443f6fafac86c1a5c04cc1</anchor>
      <arglist>(const MatrixPointer_ &amp;matrix, const ScalePointer_ &amp;scale, bool column, bool divide)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>rows</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>aee91777f1fc6065af5d62da64014317c</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>cols</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>a7b7503144b1b012eb434e40bbc7290cc</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>af3f177b829d11e61dc7637d26395657d</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_adjoint_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>abcd60a1d5a7875157155be1ab7b7f2ee</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_realize_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>a6e352efaf051c6158a02919e45c35257</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ScaledWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_scale)&gt; &gt; &gt;</type>
      <name>new_known_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>a996ab70cb7743bb6b83b9cc53901c7ee</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ScaledAdjointWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_scale)&gt; &gt; &gt;</type>
      <name>new_known_adjoint_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>ac4386c1d79868751455714559357d621</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; ScaledRealizeWorkspace&lt; EigenMatrix_, I&lt; decltype(*my_matrix)&gt;, I&lt; decltype(*my_scale)&gt; &gt; &gt;</type>
      <name>new_known_realize_workspace</name>
      <anchorfile>classirlba_1_1ScaledMatrix.html</anchorfile>
      <anchor>a10fd288ec69fb77a6fd6dcb147cd11b2</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ScaledRealizeWorkspace</name>
    <filename>classirlba_1_1ScaledRealizeWorkspace.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Scale_</templarg>
    <base>irlba::RealizeWorkspace&lt; EigenMatrix_ &gt;</base>
    <member kind="function">
      <type>const EigenMatrix_ &amp;</type>
      <name>realize</name>
      <anchorfile>classirlba_1_1ScaledRealizeWorkspace.html</anchorfile>
      <anchor>ac6b612318b238a96396008d91ee61c89</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::ScaledWorkspace</name>
    <filename>classirlba_1_1ScaledWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Matrix_</templarg>
    <templarg>class Scale_</templarg>
    <base>irlba::Workspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1ScaledWorkspace.html</anchorfile>
      <anchor>a715006cf4bc73e133803e8cab60c6e47</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;out)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::SimpleAdjointWorkspace</name>
    <filename>classirlba_1_1SimpleAdjointWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Simple_</templarg>
    <base>irlba::AdjointWorkspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1SimpleAdjointWorkspace.html</anchorfile>
      <anchor>a0d058d982265184bfcef57e33ba9b4d9</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::SimpleMatrix</name>
    <filename>classirlba_1_1SimpleMatrix.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class SimplePointer_</templarg>
    <base>irlba::Matrix&lt; EigenVector_, EigenMatrix_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>SimpleMatrix</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a40ad6109c43db9aae47c10eb731039c6</anchor>
      <arglist>(SimplePointer_ matrix)</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>rows</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a61f8ae12917ce7029309a94bcd19076a</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>Eigen::Index</type>
      <name>cols</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>abea605d297da66b88ac041d8312803c6</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; Workspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a4d828993e46ab39f37647d8d77b6b86b</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; AdjointWorkspace&lt; EigenVector_ &gt; &gt;</type>
      <name>new_adjoint_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a4722a76b2cf4c2c78131b0f0e0c96878</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; RealizeWorkspace&lt; EigenMatrix_ &gt; &gt;</type>
      <name>new_realize_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>acde47445ae7eb9e599c6fbd1b34b14ea</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; SimpleWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt; &gt; &gt;</type>
      <name>new_known_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a227b3b5937dd21b0e0150652669f9f29</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; SimpleAdjointWorkspace&lt; EigenVector_, I&lt; decltype(*my_matrix)&gt; &gt; &gt;</type>
      <name>new_known_adjoint_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>aff10a1242e8858f5223cc085b209fb24</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::unique_ptr&lt; SimpleRealizeWorkspace&lt; EigenMatrix_, I&lt; decltype(*my_matrix)&gt; &gt; &gt;</type>
      <name>new_known_realize_workspace</name>
      <anchorfile>classirlba_1_1SimpleMatrix.html</anchorfile>
      <anchor>a479e87be37b59b44dcd258bc1a0aebce</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::SimpleRealizeWorkspace</name>
    <filename>classirlba_1_1SimpleRealizeWorkspace.html</filename>
    <templarg>class EigenMatrix_</templarg>
    <templarg>class Simple_</templarg>
    <base>irlba::RealizeWorkspace&lt; EigenMatrix_ &gt;</base>
    <member kind="function">
      <type>const EigenMatrix_ &amp;</type>
      <name>realize</name>
      <anchorfile>classirlba_1_1SimpleRealizeWorkspace.html</anchorfile>
      <anchor>a2bff5fb36075a9cbb8245fc51972a590</anchor>
      <arglist>(EigenMatrix_ &amp;buffer)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::SimpleWorkspace</name>
    <filename>classirlba_1_1SimpleWorkspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <templarg>class Simple_</templarg>
    <base>irlba::Workspace&lt; EigenVector_ &gt;</base>
    <member kind="function">
      <type>void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1SimpleWorkspace.html</anchorfile>
      <anchor>ad4233a023a12e966556b72cd4f816478</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>irlba::Workspace</name>
    <filename>classirlba_1_1Workspace.html</filename>
    <templarg>class EigenVector_</templarg>
    <member kind="function" virtualness="pure">
      <type>virtual void</type>
      <name>multiply</name>
      <anchorfile>classirlba_1_1Workspace.html</anchorfile>
      <anchor>aace68ce16da70c397d4a790f3667ae35</anchor>
      <arglist>(const EigenVector_ &amp;right, EigenVector_ &amp;output)=0</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>irlba</name>
    <filename>namespaceirlba.html</filename>
    <class kind="class">irlba::AdjointWorkspace</class>
    <class kind="class">irlba::CenteredAdjointWorkspace</class>
    <class kind="class">irlba::CenteredMatrix</class>
    <class kind="class">irlba::CenteredRealizeWorkspace</class>
    <class kind="class">irlba::CenteredWorkspace</class>
    <class kind="class">irlba::EigenThreadScope</class>
    <class kind="class">irlba::Matrix</class>
    <class kind="struct">irlba::Options</class>
    <class kind="class">irlba::ParallelSparseAdjointWorkspace</class>
    <class kind="class">irlba::ParallelSparseMatrix</class>
    <class kind="class">irlba::ParallelSparseRealizeWorkspace</class>
    <class kind="class">irlba::ParallelSparseWorkspace</class>
    <class kind="class">irlba::RealizeWorkspace</class>
    <class kind="struct">irlba::Results</class>
    <class kind="class">irlba::ScaledAdjointWorkspace</class>
    <class kind="class">irlba::ScaledMatrix</class>
    <class kind="class">irlba::ScaledRealizeWorkspace</class>
    <class kind="class">irlba::ScaledWorkspace</class>
    <class kind="class">irlba::SimpleAdjointWorkspace</class>
    <class kind="class">irlba::SimpleMatrix</class>
    <class kind="class">irlba::SimpleRealizeWorkspace</class>
    <class kind="class">irlba::SimpleWorkspace</class>
    <class kind="class">irlba::Workspace</class>
    <member kind="function">
      <type>std::pair&lt; bool, int &gt;</type>
      <name>compute</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a6fe58ca810293f76a81ff7ab05d041f4</anchor>
      <arglist>(const Matrix_ &amp;matrix, Eigen::Index number, EigenMatrix_ &amp;outU, EigenMatrix_ &amp;outV, EigenVector_ &amp;outD, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>std::pair&lt; bool, int &gt;</type>
      <name>compute_simple</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>ad3aed8e6de2f1f8f82b54a84f419ded5</anchor>
      <arglist>(const InputEigenMatrix_ &amp;matrix, Eigen::Index number, OutputEigenMatrix_ &amp;outU, OutputEigenMatrix_ &amp;outV, EigenVector_ &amp;outD, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Results&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>compute</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a1fba15566b7b2f50bdcddb1b7df3ed4c</anchor>
      <arglist>(const Matrix_ &amp;matrix, Eigen::Index number, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Results&lt; OutputEigenMatrix_, EigenVector_ &gt;</type>
      <name>compute_simple</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a7f0da0f8a90c7aeb5f243097a751b87e</anchor>
      <arglist>(const InputEigenMatrix_ &amp;matrix, Eigen::Index number, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>parallelize</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a7ddcae16077f8210cfab2f59ee3effed</anchor>
      <arglist>(Task_ num_tasks, Run_ run_task)</arglist>
    </member>
    <member kind="function">
      <type>std::pair&lt; bool, int &gt;</type>
      <name>pca</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a0f28d73593e7da019e54b134dd149888</anchor>
      <arglist>(const InputEigenMatrix_ &amp;matrix, bool center, bool scale, Eigen::Index number, OutputEigenMatrix_ &amp;outU, OutputEigenMatrix_ &amp;outV, EigenVector_ &amp;outD, const Options &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Results&lt; OutputEigenMatrix_, EigenVector_ &gt;</type>
      <name>pca</name>
      <anchorfile>namespaceirlba.html</anchorfile>
      <anchor>a4a834ea1fa5dd1463d27f407219deb19</anchor>
      <arglist>(const InputEigenMatrix_ &amp;matrix, bool center, bool scale, Eigen::Index number, const Options &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ library for IRLBA</title>
    <filename>index.html</filename>
    <docanchor file="index.html" title="C++ library for IRLBA">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
