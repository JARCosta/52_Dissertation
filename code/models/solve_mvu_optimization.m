function [G, cvx_status] = solve_mvu_optimization(n, NG, eps)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    % X: n-by-D data matrix (rows are points)
    % NG: sparse matrix of neighborhood graph    

    format shortG

    [i_indices, j_indices] = find(NG > 0);
    
    distances = NG(sub2ind([n,n], i_indices, j_indices));

    % inner_prod = X * X';
    
    % ratio = round(log10(max(inner_prod(:)))) - 2;
    % ratio = 10^(ratio);
    % disp(ratio)
    % inner_prod = inner_prod * ratio;
    
    % inner_prod_diag = diag(inner_prod);
    % distances_inner = inner_prod_diag(i_indices) + inner_prod_diag(j_indices) - 2*inner_prod(sub2ind([n,n], i_indices, j_indices));
    % % disp(max(distances_inner))
    % distances = distances_inner;
    
    % if length(distances_inner) < 100
    %     disp("distances:")
    %     disp(distances)
    %     disp("distances_inner:")
    %     disp(distances_inner)
    %     disp("distances - distances_inner:")
    %     disp(distances - distances_inner)
    % end
    
    % D = pdist2(X, X).^2; % squared pairwise distances
    % disp(eps)
    % eps = 1e-3;

    % ==== Step 3: Solve MVU via CVX ====
    cvx_begin sdp quiet
        cvx_solver mosek

        if eps ~= 1e-8
            cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_PFEAS', eps)
            cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_DFEAS', eps)
            cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_REL_GAP', eps)
            cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_INFEAS', eps * 0.01)
            
            cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_NEAR_REL', 1/eps)
        end

        variable G(n, n) symmetric
        maximize( trace(G) )
        subject to
            G >= 0;
            sum(G(:)) == 0;
            % trace(G) <= n;
            % norm(G, 'fro') <= sqrt(n);
            
            G_diag = diag(G);
            gram_distances = G_diag(i_indices) + G_diag(j_indices) - 2*G(sub2ind([n,n], i_indices, j_indices));
            
            gram_distances == distances;
            % gram_distances == D(sub2ind([n,n], i_indices, j_indices));
        
    cvx_end
    % if ratio ~= 1
    %     G = G / ratio;
    % end
end 