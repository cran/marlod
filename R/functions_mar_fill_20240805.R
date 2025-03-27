###Function for modified GEE

Modified.GEE=function(id, y, x, lod, substitue, corstr, typetd, maxiter)
{
  #############################################################
  # "impute.boot" function from "miWQS" package (version 0.4.4)
  #############################################################
  impute.boot=function(X, DL, Z = NULL, K = 5L, verbose = FALSE)
  {
    check_function.Lub=function(chemcol, dlcol, Z, K, verbose)
    {
      X <- check_X(as.matrix(chemcol))
      if (ncol(X) > 1) {
        warning("This approach cannot impute more than one chemical at time. Imputing first element...")
        chemcol <- chemcol[1]
      }
      else {
        chemcol <- as.vector(X)
      }
      if (is.null(dlcol))
        stop("dlcol is NULL. A detection limit is needed", call. = FALSE)
      if (is.na(dlcol))
        stop("The detection limit has missing values so chemical is not imputed.",
             immediate. = FALSE, call. = FALSE)
      if (!is.numeric(dlcol))
        stop("The detection limit is non-numeric. Detection limit must be numeric.",
             call. = FALSE)
      if (!is.vector(dlcol))
        stop("The detection limit must be a vector.", call. = FALSE)
      if (length(dlcol) > 1) {
        if (min(dlcol, na.rm = TRUE) == max(dlcol, na.rm = TRUE)) {
          dlcol <- unique(dlcol)
        }
        else {
          warning(" The detection limit is not unique, ranging from ",
                  min(dlcol, na.rm = TRUE), " to ", max(dlcol,
                                                        na.rm = TRUE), "; The smallest value is assumed to be detection limit",
                  call. = FALSE, immediate. = FALSE)
          detcol <- !is.na(chemcol)
          if (verbose) {
          cat("\n Summary when chemical is missing (BDL) \n")
          #print(summary(dlcol[detcol == 0]))
          cat("## when chemical is observed \n ")
          #print(summary(dlcol[detcol == 1]))
          dlcol <- min(dlcol, na.rm = TRUE)
          }
        }
      }
      if (is.null(Z)) {
        Z <- matrix(rep(1, length(chemcol)), ncol = 1)
      }
      if (anyNA(Z)) {
        warning("Missing covariates are ignored.")
      }
      if (!is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      K <- check_constants(K)
      return(list(chemcol = chemcol, dlcol = dlcol, Z = Z, K = K))
    } #End of check_function.Lub

    impute.Lubin=function(chemcol, dlcol, Z = NULL, K = 5L, verbose = FALSE)
    {
      l <- check_function.Lub(chemcol, dlcol, Z, K, verbose)
      dlcol <- l$dlcol
      Z <- l$Z
      K <- l$K
      n <- length(chemcol)
      chemcol2 <- ifelse(is.na(chemcol), 1e-05, chemcol)
      event <- ifelse(is.na(chemcol), 3, 1)
      fullchem_data <- na.omit(data.frame(id = seq(1:n), chemcol2 = chemcol2,
                                          event = event, LB = rep(0, n), UB = rep(dlcol, n), Z))
      n <- nrow(fullchem_data)
      if (verbose) {
        cat("Dataset Used in Survival Model \n")
        #print(head(fullchem_data))
      }
      bootstrap_data <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      beta_not <- NA
      std <- rep(0, K)
      unif_lower <- rep(0, K)
      unif_upper <- rep(0, K)
      imputed_values <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      for (a in 1:K) {
        bootstrap_data[, a] <- as.vector(sample(1:n, replace = TRUE))
        data_sample <- fullchem_data[bootstrap_data[, a], ]
        freqs <- as.data.frame(table(bootstrap_data[, a]))
        freqs_ids <- as.numeric(as.character(freqs[, 1]))
        my.weights <- freqs[, 2]
        final <- fullchem_data[freqs_ids, ]
        my.surv.object <- survival::Surv(time = final$chemcol2,
                                         time2 = final$UB, event = final$event, type = "interval")
        model <- survival::survreg(my.surv.object ~ ., data = final[,
                                                                    -(1:5), drop = FALSE], weights = my.weights, dist = "lognormal",
                                   x = TRUE)
        beta_not <- rbind(beta_not, model$coefficients)
        std[a] <- model$scale
        Z <- model$x
        mu <- Z %*% beta_not[a + 1, ]
        unif_lower <- plnorm(1e-05, mu, std[a])
        unif_upper <- plnorm(dlcol, mu, std[a])
        u <- runif(n, unif_lower, unif_upper)
        imputed <- qlnorm(u, mu, std[a])
        imputed_values[, a] <- ifelse(fullchem_data$chemcol2 ==
                                        1e-05, imputed, chemcol)
      }
      x.miss.index <- ifelse(chemcol == 0 | is.na(chemcol), TRUE,
                             FALSE)
      indicator.miss <- sum(imputed_values[x.miss.index, ] > dlcol)
      if (verbose) {
        beta_not <- beta_not[-1, , drop = FALSE]
        cat("\n ## MLE Estimates \n")
        A <- round(cbind(beta_not, std), digits = 4)
        colnames(A) <- c(names(model$coefficients), "stdev")
        #print(A)
        cat("## Uniform Imputation Range \n")
        B <- rbind(format(range(unif_lower), digits = 3, scientific = TRUE),
                   format(range(unif_upper), digits = 3, nsmall = 3))
        rownames(B) <- c("unif_lower", "unif_upper")
        #print(B)
        cat("## Detection Limit:", unique(dlcol), "\n")
      }
      bootstrap_data <- apply(bootstrap_data, 2, as.factor)
      return(list(imputed_values = imputed_values, bootstrap_index = bootstrap_data,
                  indicator.miss = indicator.miss))
    } #End of impute.Lubin

    check_covariates=function(Z, X)
    {
      nZ <- if (is.vector(Z) | is.factor(Z)) {
        length(Z)
      }
      else {
        nrow(Z)
      }
      if (nrow(X) != nZ) {
        if (verbose) {
        cat("> X has", nrow(X), "individuals, but Z has", nZ,
            "individuals")
        stop("Can't Run. The total number of individuals with components X is different than total number of individuals with covariates Z.",
             call. = FALSE)
        }
      }
      if (anyNA(Z)) {
        p <- if (is.vector(Z) | is.factor(Z)) {
          1
        }
        else {
          ncol(Z)
        }
        index <- complete.cases(Z)
        Z <- Z[index, , drop = FALSE]
        X <- X[index, , drop = FALSE]
        dimnames(X)[[1]] <- 1:nrow(X)
        warning("Covariates are missing for individuals ", paste(which(!index),
                                                                 collapse = ", "), " and are ignored. Subjects renumbered in complete data.",
                call. = FALSE, immediate. = TRUE)
      }
      if (!is.null(Z) & !is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else if (is.array(Z)) {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      return(list(Z = Z, X = X))
    } #End of check_covariates

    is.wholenumber=function(x)
    {
      x > -1 && is.integer(x)
    } #End of is.wholenumber
    #is.wholenumber=function(x, tol = .Machine$double.eps^0.5)
    #{
    #  x > -1 && is.integer(x, tol)
    #} #End of is.wholenumber

    check_constants=function(K)
    {
      if (is.null(K))
        stop(sprintf(" must be non-null."), call. = TRUE)
      if (!is.numeric(K) | length(K) != 1)
        stop(sprintf(" must be numeric of length 1"), call. = TRUE)
      if (K <= 0)
        stop(sprintf(" must be a positive integer"), call. = TRUE)
      if (!is.wholenumber(K)) {
        K <- ceiling(K)
      }
      return(K)
    } #End of check_constants

    check_X=function(X)
    {
      if (is.null(X)) {
        stop("X is null. X must have some data.", call. = FALSE)
      }
      if (is.data.frame(X)) {
        X <- as.matrix(X)
      }
      if (!all(apply(X, 2, is.numeric))) {
        stop("X must be numeric. Some columns in X are not numeric.", call. = FALSE)
      }
      return(X)
    } #End of check_X

    check_imputation=function(X, DL, Z, K, T, n.burn, verbose = NULL)
    {
      if (is.vector(X)) {
        X <- as.matrix(X)
      }
      X <- check_X(X)
      if (!anyNA(X))
        stop("Matrix X has nothing to impute. No need to impute.", call. = FALSE)
      if (is.matrix(DL) | is.data.frame(DL)) {
        if (ncol(DL) == 1 | nrow(DL) == 1) {
          DL <- as.numeric(DL)
        }
        else {
          stop("The detection limit must be a vector.", call. = FALSE)
        }
      }
      stopifnot(is.numeric(DL))
      if (anyNA(DL)) {
        no.DL <- which(is.na(DL))
        warning("The detection limit for ", names(no.DL), " is missing.\n Both the chemical and detection limit is removed in imputation.",
                call. = FALSE)
        DL <- DL[-no.DL]
        if (!(names(no.DL) %in% colnames(X))) {
          if (verbose) {
          cat("  ##Missing DL does not match colnames(X) \n")
          }
        }
        X <- X[, -no.DL]
      }
      if (length(DL) != ncol(X)) {
        if (verbose) {
        cat("The following components \n")
        }
        X[1:3, ]
        if (verbose) {
        cat("have these detection limits \n")
        }
        DL
        stop("Each component must have its own detection limit.", call. = FALSE)
      }
      K <- check_constants(K)
      if (!is.null(n.burn) & !is.null(T)) {
        stopifnot(is.wholenumber(n.burn))
        T <- check_constants(T)
        if (!(n.burn < T))
          stop("Burn-in is too large; burn-in must be smaller than MCMC iterations T", call. = FALSE)
      }
      return(list(X = X, DL = DL, Z = Z, K = K, T = T))
    } #End of check_imputation

    check <- check_imputation(X = X, DL = DL, Z = Z, K = K, T = 5, n.burn = 4L, verbose)
    X <- check$X
    DL <- check$DL
    if (is.null(Z)) {
      Z <- matrix(1, nrow = nrow(X), ncol = 1, dimnames = list(NULL, "Intercept"))
    }
    else {
      l <- check_covariates(Z, X)
      X <- l$X
      Z <- l$Z
    }
    K <- check$K
    if (verbose) {
      cat("X \n")
      #print(summary(X))
      cat("DL \n")
      #print(DL)
      cat("Z \n")
      #print(summary(Z))
      cat("K =", K, "\n")
    }
    n <- nrow(X)
    c <- ncol(X)
    chemical.name <- colnames(X)
    results_Lubin <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    bootstrap_index <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    indicator.miss <- rep(NA, c)
    for (j in 1:c) {
      answer <- impute.Lubin(chemcol = X[, j], dlcol = DL[j], Z = Z, K = K)
      results_Lubin[, j, ] <- as.array(answer$imputed_values, dim = c(n, j, K))
      bootstrap_index[, j, ] <- array(answer$bootstrap_index, dim = c(n, 1, K))
      indicator.miss[j] <- answer$indicator.miss
    }
    total.miss <- sum(indicator.miss)
    if (total.miss == 0) {
      #message("#> Check: The total number of imputed values that are above the detection limit is ", sum(indicator.miss), ".")
    }
    else {
      #print(indicator.miss)
      stop("#> Error: Incorrect imputation is performed; some imputed values are above the detection limit. Could some of `X` have complete data under a lower bound (`DL`)?", call. = FALSE)
    }
    return(list(X.imputed = results_Lubin, bootstrap_index = answer$bootstrap_index, indicator.miss = total.miss))
  }
  ######################
  # End of "impute.boot"
  ######################

  fam = "gaussian" #Family for the outcomes
  obs=lapply(split(id,id),"length")
  nobs <- as.numeric(obs)             #Vector of cluster sizes for each cluster
  nsub <- length(nobs)                #Number of clusters, n or N

  #Substitution method using LOD
  if(substitue=="None") cen_y <- y

  #Substitution method using LOD
  if(substitue=="LOD") cen_y <- ifelse(y>=lod,y,lod)

  #Substitution method using LOD/2
  if(substitue=="LOD2") cen_y <- ifelse(y>=lod,y,lod/2)

  #Substitution method using LOD/square root of 2
  if(substitue=="LODS2") cen_y <- ifelse(y>=lod,y,lod/sqrt(2))

  #Beta-substitution method using mean
  if(substitue=="BetaMean"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_mean <- length(y)/(length(y)-length(y1))*pnorm(z-sy)*exp(-sy*z+(sy)^2/2)
    cen_y <- ifelse(y>=lod, y, lod*beta_mean)
  }
  #Beta-substitution method using geometric mean
  if(substitue=="BetaGM"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_GM <- exp((-(length(y)-(length(y)-length(y1)))*length(y))/(length(y)-length(y1))*log(f_sy_z)-sy*z-(length(y)-(length(y)-length(y1)))/(2*(length(y)-length(y1))*length(y))*(sy)^2)
    cen_y <- ifelse(y>=lod, y, lod*beta_GM)
  }
  #Multiple imputation method with one covariate using id
  if(substitue=="MIWithID"){
    y1 <- ifelse(y>=lod,y,NA)
    results_boot2 <- impute.boot(X=y1, DL=lod, Z=id, K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot2$X.imputed, 1, mean))
  }
  #Multiple imputation method with two covariates using id and visit
  if(substitue=="MIWithIDRM"){
    y1 <- ifelse(y>=lod,y,NA)
    Z2=NULL
    for(i in 1:nsub){
      Z1=as.matrix(1:nobs[i])
      Z2=rbind(Z2,Z1)
    }
    results_boot3 <- impute.boot(X=y1, DL=lod, Z=cbind(id,Z2), K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot3$X.imputed, 1, mean))
  }
  #Multiple imputation method using QQ-plot approach
  if(substitue=="QQplot"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)

    obs <- rank(y)
    rank <- (obs-0.5)/length(y)
    zscore0 <- qnorm(rank)*ifelse(y>=lod,1,0)
    zscore1 <- zscore0[which(zscore0!=0)]

    data_frame0=data.frame(cbind(lny1,zscore1))
    beta_est=as.matrix(glm(lny1~zscore1, data=data_frame0, family=gaussian)$coefficients)
    lny_zscore <- beta_est[1,1] + beta_est[2,1]*qnorm(rank)
    y_zscore <- exp(lny_zscore)
    cen_y <- ifelse(y>=lod,y,y_zscore)
  }
  y <- cen_y

  tol=0.00000001 #Used to determine convergence

  if (corstr == "exchangeable" )
  {
    N_star = 0
    for(i in 1:nsub)
      N_star = N_star + 0.5*nobs[i]*(nobs[i]-1)  #Used in estimating the correlation
  }

  if (corstr == "AR-1" )   K_1 = sum(nobs) - nsub

  one=matrix(1,length(id),1)
  x=cbind(one,x)                   #Design matrix

  np=dim(x)[2]                     #Number of regression parameters (betas)
  type=as.matrix(c(1,typetd))      #Time-dependent covariate types

  Model_basedVar=matrix(0,np,np)   #Initial estimate of the asymptotic variance matrix.  Needed in order to calculate H for Mancl and DeRouen's (2001) method

  zero=matrix(0,np,np)

  beta=matrix(0,np,1)

  betadiff <- 1
  iteration <- 0
  betanew <- beta

  if ((corstr == "exchangeable") | (corstr == "AR-1"))
    corr_par = 0  #This is the correlation parameter, given an initial value of 0


  scale=0      #Used for the correlation estimation.
  N=sum(nobs)  #Used in estimating the scale parameter.


  while (betadiff > tol && iteration < maxiter)
  {
    beta <- betanew
    GEE=matrix(0,np,1)       #This will be the summation of the n components of the GEE.
    I=matrix(0,np,np)        #This will be the sum of the components in the variance equation
    #wi=matrix(0,np,ni)
    I_ind=matrix(0,np,np)    #Used for CIC: model-based covariance matrix assuming independence

    if ((corstr == "exchangeable") | (corstr == "AR-1")) sum = 0

    sum_scale=0

    meat=0                 #This is the "meat" of the empirical variance estimate
    meat_BC_MD=meat_BC_KC=0   #This is the "meat" of the bias-corrected empirical variance estimates


    loc1 <- 0
    loc2 <- 0
    for(i in 1:nsub){
      loc1 <- loc2 + 1
      loc2 <- loc1 + nobs[i] - 1
      yi <- as.matrix(y[loc1:loc2, ])
      xi <- x[loc1:loc2, ]
      ni <- nrow(yi)
      wi=matrix(0,np,ni)
      if(ni==1)
        xi=t(xi)

      #Note that we are obtaining the Pearson residuals in here, which are used to estimate the correlation parameter.
      if (fam == "gaussian") {
        ui <- xi %*% beta
        fui <- ui
        fui_dev <- diag(ni)
        vui <- diag(ni)
        Pearson = (yi-ui)
      }

      #Now for obtaining the sums (from each independent subject or cluster) used in estimating the correlation parameter and also scale parameter.

      if (ni > 1) {
      for(j in 1:ni)
        sum_scale = sum_scale + Pearson[j,1]^2

      if (corstr == "exchangeable") {
        for(j in 1:(ni-1))
          for(k in (j+1):ni)
            sum = sum + Pearson[j,1]*Pearson[k,1]
      }

      if (corstr == "AR-1") {
        for(j in 1:(ni-1))
          sum = sum + Pearson[j,1]*Pearson[(j+1),1]
      }
      }

      #Now for obtaining the sums used in the GEE for the betas


      ################################################################################
      if (corstr == "exchangeable") {

        Di = fui_dev %*% xi
        DiT = t(Di)

        Ri = diag(ni) + matrix(corr_par,ni,ni) - diag(corr_par,ni)     #This is the current estimated working correlation matrix
        Ri_inv = ginv(Ri)

        R_Type1_inv = Ri_inv

        Ri_inv2 = Ri_inv
        Ri_inv2[upper.tri(Ri_inv2)] = 0
        R_Type2_inv = Ri_inv2

        Ri_inv3 = Ri_inv
        Ri_inv3[upper.tri(Ri_inv3)] = 0
        Ri_inv3[lower.tri(Ri_inv3)] = 0
        R_Type3_inv = Ri_inv3

        Ri_inv4 = Ri_inv
        Ri_inv4[lower.tri(Ri_inv4)] = 0
        R_Type4_inv = Ri_inv4

        for(j in 1:np)
        {
          if(type[j,1]==1){
            wi[j,] = DiT[j,] %*% vui %*% R_Type1_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==2){
            wi[j,] = DiT[j,] %*% vui %*% R_Type2_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==3){
            wi[j,] = DiT[j,] %*% vui %*% R_Type3_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==4){
            wi[j,] = DiT[j,] %*% vui %*% R_Type4_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
        }
      }

      ################################################################################
      if (corstr == "AR-1") {

        Di = fui_dev %*% xi
        DiT = t(Di)

        Ri = matrix(0,ni,ni)                          #This is the current estimated working correlation matrix
        for(k in 1:ni)
          for(l in 1:ni)
            Ri[k,l]=corr_par^abs(k-l)

        Ri_inv = ginv(Ri)

        R_Type1_inv = Ri_inv

        Ri_inv2 = Ri_inv
        Ri_inv2[upper.tri(Ri_inv2)] = 0
        R_Type2_inv = Ri_inv2

        Ri_inv3 = Ri_inv
        Ri_inv3[upper.tri(Ri_inv3)] = 0
        Ri_inv3[lower.tri(Ri_inv3)] = 0
        R_Type3_inv = Ri_inv3

        Ri_inv4 = Ri_inv
        Ri_inv4[lower.tri(Ri_inv4)] = 0
        R_Type4_inv = Ri_inv4

        for(j in 1:np)
        {
          if(type[j,1]==1){
            wi[j,] = DiT[j,] %*% vui %*% R_Type1_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==2){
            wi[j,] = DiT[j,] %*% vui %*% R_Type2_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==3){
            wi[j,] = DiT[j,] %*% vui %*% R_Type3_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
          if(type[j,1]==4){
            wi[j,] = DiT[j,] %*% vui %*% R_Type4_inv %*% vui
            GEE[j,] = GEE[j,] + wi[j,] %*% (yi - ui)
            I[j,] = I[j,] + wi[j,] %*% Di
          }
        }
      }

      I_ind = I_ind + t(xi) %*% fui_dev %*%  vui %*% vui %*% fui_dev %*% xi  #For the CIC
      #################################################################################

      H = Di%*%Model_basedVar%*%wi
      meat = meat + wi %*% (yi - ui) %*% t((yi - ui)) %*% t(wi)
      meat_BC_MD = meat_BC_MD + wi %*% ginv(diag(ni)-H) %*% (yi - ui) %*% t((yi - ui)) %*% ginv(diag(ni)-t(H)) %*% t(wi)      #Mancl and DeRouen (2001)
      meat_BC_KC = meat_BC_KC + wi %*% ginv(diag(ni)-H) %*% (yi - ui) %*% t((yi - ui)) %*% t(wi)                               #Kauermann and Carroll (2001)


    } #end of the for loop (so done adding up the GEE and I and residual terms for each independent subject)


    betanew <- beta + ginv(I) %*% GEE

    betadiff <- sum(abs(betanew - beta))                           #Determining if convergence or not

    Model_basedVar = ginv(I)                                      #The current model-based variance estimate, to be used to obtain H above

    #Updating the estimate of the correlation parameter

    scale=sum_scale/(N-np)

    if (corstr == "exchangeable")
      corr_par = sum/(N_star-np)/scale

    if (corstr == "AR-1")
      corr_par = sum/(K_1-np)/scale

    if (corstr == "unstructured")
      corr_par = sum/(nsub-np)/scale


    #Now getting the empirical variances

    emp_var = ginv(I) %*% meat %*% ginv(I)
    covariance_MD = ginv(I) %*% meat_BC_MD %*% ginv(I)         #Mancl and DeRouen (2001)
    covariance_KC = ginv(I) %*% meat_BC_KC %*% ginv(I)         #Kauermann and Carroll (2001)
    covariance_AVG = (covariance_MD + covariance_KC)/2           #Average of Mancl and DeRouen (2001) and Kauermann and Carroll (2001)

    iteration <- iteration + 1                                   #Counting how many iterations it has gone through

  } #End of the while loop

  I_ind = I_ind/scale

  beta_number=as.matrix(seq(1:np))-1 #Used to indicate the parameter. For instance, 0 denotes the intercept.
  SE_A=SE_MD=SE_KC=SE_AVG=TECM_A=TECM_MD=TECM_KC=TECM_AVG=CIC_A=CIC_MD=CIC_KC=CIC_AVG=matrix(0,np,1)
  for(j in 1:np){
    SE_A[j,1]=sqrt(emp_var[j,j])
    SE_MD[j,1]=sqrt(covariance_MD[j,j])
    SE_KC[j,1]=sqrt(covariance_KC[j,j])
    SE_AVG[j,1]=sqrt(covariance_AVG[j,j])}
  Wald_A=Wald_MD=Wald_KC=Wald_AVG=matrix(0,np,1)
  for(j in 1:np){
    Wald_A[j,1]=beta[j,1]/SE_A[j,1]
    Wald_MD[j,1]=beta[j,1]/SE_MD[j,1]
    Wald_KC[j,1]=beta[j,1]/SE_KC[j,1]
    Wald_AVG[j,1]=beta[j,1]/SE_AVG[j,1]}
  df=nsub-np
  p_valueA=p_valueMD=p_valueKC=p_valueAVG=matrix(0,np,1)
  for(j in 1:np){
    p_valueA[j,1]=1-pf((Wald_A[j,1])^2,1,df)
    p_valueMD[j,1]=1-pf((Wald_MD[j,1])^2,1,df)
    p_valueKC[j,1]=1-pf((Wald_KC[j,1])^2,1,df)
    p_valueAVG[j,1]=1-pf((Wald_AVG[j,1])^2,1,df)}

  TECM_A[1,1]=sum(diag( emp_var ))
  TECM_MD[1,1]=sum(diag( covariance_MD ))
  TECM_KC[1,1]=sum(diag( covariance_KC ))
  TECM_AVG[1,1]=sum(diag( covariance_KC ))
  CIC_A[1,1]=sum(diag( I_ind %*% emp_var ))
  CIC_MD[1,1]=sum(diag( I_ind %*% covariance_MD ))
  CIC_KC[1,1]=sum(diag( I_ind %*% covariance_KC ))
  CIC_AVG[1,1]=sum(diag( I_ind %*% covariance_AVG ))

  output_A=cbind(beta_number,beta,SE_A,Wald_A,p_valueA,TECM_A,CIC_A)
  output_MD=cbind(beta_number,beta,SE_MD,Wald_MD,p_valueMD,TECM_MD,CIC_MD)
  output_KC=cbind(beta_number,beta,SE_KC,Wald_KC,p_valueKC,TECM_KC,CIC_KC)
  output_AVG=cbind(beta_number,beta,SE_AVG,Wald_AVG,p_valueAVG,TECM_AVG,CIC_AVG)

  colnames(output_A) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_MD) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_KC) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_AVG) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")

  correction_A=correction_MD=correction_KC=correction_AVG=Parameter1=Parameter2=matrix(0,1,7)
  correction_A[1,]=c("Typical Asymptotic SE","","","","","","")
  correction_MD[1,]=c("Bias-Corrected SE (MD)","","","","","","")
  correction_KC[1,]=c("Bias-Corrected SE (KC)","","","","","","")
  correction_AVG[1,]=c("Bias-Corrected SE (AVG)","","","","","","")
  Parameter1[1,1]="Estimated correlation"
  Parameter1[1,2]=corr_par
  Parameter2[1,1]="Estimated dispersion"
  Parameter2[1,2]=scale
  output_ALL=rbind(correction_A,output_A,correction_MD,output_MD,correction_KC,output_KC,correction_AVG,output_AVG,Parameter1,Parameter2)

  message("Results include typical asymptotic standard error (SE) estimates and bias-corrected SE estimates that use covariance inflation corrections.")
  message("Covariance inflation corrections are based on the corrections of Mancl and DeRouen (MD) (2001), and Kauermann and Carroll (KC) (2001).")
  message("Last covariance inflation correction averages the above corrections (AVG) (Ford and Westgate, 2017, 2018).")
  return(knitr::kable(output_ALL))

  #message("Results based on the typical asymptotic standard error estimates")
  #message("Results based on the bias-corrected standard error estimates that use the covariance inflation correction and the correction that is based on the Mancl and DeRouen (2001) correction")
  #message("Results based on the bias-corrected standard error estimates that use the covariance inflation correction and the correction that is based on the Kauermann and Carroll (2001) correction")
  #message("Results based on the bias-corrected standard error estimates that use the covariance inflation correction and the correction that averages the above corrections (Ford and Westgate, 2017, 2018)")
  #message("Estimated correlation parameter:")
  #message("Estimated common dispersion parameter:")
}










###Functions for modified QIF

Modified.QIF=function(id, y, x, lod, substitue, corstr, beta, typetd, maxiter)
{
  #############################################################
  # "impute.boot" function from "miWQS" package (version 0.4.4)
  #############################################################
  impute.boot=function(X, DL, Z = NULL, K = 5L, verbose = FALSE)
  {
    check_function.Lub=function(chemcol, dlcol, Z, K, verbose)
    {
      X <- check_X(as.matrix(chemcol))
      if (ncol(X) > 1) {
        warning("This approach cannot impute more than one chemical at time. Imputing first element...")
        chemcol <- chemcol[1]
      }
      else {
        chemcol <- as.vector(X)
      }
      if (is.null(dlcol))
        stop("dlcol is NULL. A detection limit is needed", call. = FALSE)
      if (is.na(dlcol))
        stop("The detection limit has missing values so chemical is not imputed.",
             immediate. = FALSE, call. = FALSE)
      if (!is.numeric(dlcol))
        stop("The detection limit is non-numeric. Detection limit must be numeric.",
             call. = FALSE)
      if (!is.vector(dlcol))
        stop("The detection limit must be a vector.", call. = FALSE)
      if (length(dlcol) > 1) {
        if (min(dlcol, na.rm = TRUE) == max(dlcol, na.rm = TRUE)) {
          dlcol <- unique(dlcol)
        }
        else {
          warning(" The detection limit is not unique, ranging from ",
                  min(dlcol, na.rm = TRUE), " to ", max(dlcol,
                                                        na.rm = TRUE), "; The smallest value is assumed to be detection limit",
                  call. = FALSE, immediate. = FALSE)
          detcol <- !is.na(chemcol)
          if (verbose) {
            cat("\n Summary when chemical is missing (BDL) \n")
            #print(summary(dlcol[detcol == 0]))
            cat("## when chemical is observed \n ")
            #print(summary(dlcol[detcol == 1]))
            dlcol <- min(dlcol, na.rm = TRUE)
          }
        }
      }
      if (is.null(Z)) {
        Z <- matrix(rep(1, length(chemcol)), ncol = 1)
      }
      if (anyNA(Z)) {
        warning("Missing covariates are ignored.")
      }
      if (!is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      K <- check_constants(K)
      return(list(chemcol = chemcol, dlcol = dlcol, Z = Z, K = K))
    } #End of check_function.Lub

    impute.Lubin=function(chemcol, dlcol, Z = NULL, K = 5L, verbose = FALSE)
    {
      l <- check_function.Lub(chemcol, dlcol, Z, K, verbose)
      dlcol <- l$dlcol
      Z <- l$Z
      K <- l$K
      n <- length(chemcol)
      chemcol2 <- ifelse(is.na(chemcol), 1e-05, chemcol)
      event <- ifelse(is.na(chemcol), 3, 1)
      fullchem_data <- na.omit(data.frame(id = seq(1:n), chemcol2 = chemcol2,
                                          event = event, LB = rep(0, n), UB = rep(dlcol, n), Z))
      n <- nrow(fullchem_data)
      if (verbose) {
        cat("Dataset Used in Survival Model \n")
        #print(head(fullchem_data))
      }
      bootstrap_data <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      beta_not <- NA
      std <- rep(0, K)
      unif_lower <- rep(0, K)
      unif_upper <- rep(0, K)
      imputed_values <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      for (a in 1:K) {
        bootstrap_data[, a] <- as.vector(sample(1:n, replace = TRUE))
        data_sample <- fullchem_data[bootstrap_data[, a], ]
        freqs <- as.data.frame(table(bootstrap_data[, a]))
        freqs_ids <- as.numeric(as.character(freqs[, 1]))
        my.weights <- freqs[, 2]
        final <- fullchem_data[freqs_ids, ]
        my.surv.object <- survival::Surv(time = final$chemcol2,
                                         time2 = final$UB, event = final$event, type = "interval")
        model <- survival::survreg(my.surv.object ~ ., data = final[,
                                                                    -(1:5), drop = FALSE], weights = my.weights, dist = "lognormal",
                                   x = TRUE)
        beta_not <- rbind(beta_not, model$coefficients)
        std[a] <- model$scale
        Z <- model$x
        mu <- Z %*% beta_not[a + 1, ]
        unif_lower <- plnorm(1e-05, mu, std[a])
        unif_upper <- plnorm(dlcol, mu, std[a])
        u <- runif(n, unif_lower, unif_upper)
        imputed <- qlnorm(u, mu, std[a])
        imputed_values[, a] <- ifelse(fullchem_data$chemcol2 ==
                                        1e-05, imputed, chemcol)
      }
      x.miss.index <- ifelse(chemcol == 0 | is.na(chemcol), TRUE,
                             FALSE)
      indicator.miss <- sum(imputed_values[x.miss.index, ] > dlcol)
      if (verbose) {
        beta_not <- beta_not[-1, , drop = FALSE]
        cat("\n ## MLE Estimates \n")
        A <- round(cbind(beta_not, std), digits = 4)
        colnames(A) <- c(names(model$coefficients), "stdev")
        #print(A)
        cat("## Uniform Imputation Range \n")
        B <- rbind(format(range(unif_lower), digits = 3, scientific = TRUE),
                   format(range(unif_upper), digits = 3, nsmall = 3))
        rownames(B) <- c("unif_lower", "unif_upper")
        #print(B)
        cat("## Detection Limit:", unique(dlcol), "\n")
      }
      bootstrap_data <- apply(bootstrap_data, 2, as.factor)
      return(list(imputed_values = imputed_values, bootstrap_index = bootstrap_data,
                  indicator.miss = indicator.miss))
    } #End of impute.Lubin

    check_covariates=function(Z, X)
    {
      nZ <- if (is.vector(Z) | is.factor(Z)) {
        length(Z)
      }
      else {
        nrow(Z)
      }
      if (nrow(X) != nZ) {
        if (verbose) {
          cat("> X has", nrow(X), "individuals, but Z has", nZ,
              "individuals")
          stop("Can't Run. The total number of individuals with components X is different than total number of individuals with covariates Z.",
               call. = FALSE)
        }
      }
      if (anyNA(Z)) {
        p <- if (is.vector(Z) | is.factor(Z)) {
          1
        }
        else {
          ncol(Z)
        }
        index <- complete.cases(Z)
        Z <- Z[index, , drop = FALSE]
        X <- X[index, , drop = FALSE]
        dimnames(X)[[1]] <- 1:nrow(X)
        warning("Covariates are missing for individuals ", paste(which(!index),
                                                                 collapse = ", "), " and are ignored. Subjects renumbered in complete data.",
                call. = FALSE, immediate. = TRUE)
      }
      if (!is.null(Z) & !is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else if (is.array(Z)) {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      return(list(Z = Z, X = X))
    } #End of check_covariates

    is.wholenumber=function(x)
    {
      x > -1 && is.integer(x)
    } #End of is.wholenumber
    #is.wholenumber=function(x, tol = .Machine$double.eps^0.5)
    #{
    #  x > -1 && is.integer(x, tol)
    #} #End of is.wholenumber

    check_constants=function(K)
    {
      if (is.null(K))
        stop(sprintf(" must be non-null."), call. = TRUE)
      if (!is.numeric(K) | length(K) != 1)
        stop(sprintf(" must be numeric of length 1"), call. = TRUE)
      if (K <= 0)
        stop(sprintf(" must be a positive integer"), call. = TRUE)
      if (!is.wholenumber(K)) {
        K <- ceiling(K)
      }
      return(K)
    } #End of check_constants

    check_X=function(X)
    {
      if (is.null(X)) {
        stop("X is null. X must have some data.", call. = FALSE)
      }
      if (is.data.frame(X)) {
        X <- as.matrix(X)
      }
      if (!all(apply(X, 2, is.numeric))) {
        stop("X must be numeric. Some columns in X are not numeric.", call. = FALSE)
      }
      return(X)
    } #End of check_X

    check_imputation=function(X, DL, Z, K, T, n.burn, verbose = NULL)
    {
      if (is.vector(X)) {
        X <- as.matrix(X)
      }
      X <- check_X(X)
      if (!anyNA(X))
        stop("Matrix X has nothing to impute. No need to impute.", call. = FALSE)
      if (is.matrix(DL) | is.data.frame(DL)) {
        if (ncol(DL) == 1 | nrow(DL) == 1) {
          DL <- as.numeric(DL)
        }
        else {
          stop("The detection limit must be a vector.", call. = FALSE)
        }
      }
      stopifnot(is.numeric(DL))
      if (anyNA(DL)) {
        no.DL <- which(is.na(DL))
        warning("The detection limit for ", names(no.DL), " is missing.\n Both the chemical and detection limit is removed in imputation.",
                call. = FALSE)
        DL <- DL[-no.DL]
        if (!(names(no.DL) %in% colnames(X))) {
          if (verbose) {
            cat("  ##Missing DL does not match colnames(X) \n")
          }
        }
        X <- X[, -no.DL]
      }
      if (length(DL) != ncol(X)) {
        if (verbose) {
          cat("The following components \n")
        }
        X[1:3, ]
        if (verbose) {
          cat("have these detection limits \n")
        }
        DL
        stop("Each component must have its own detection limit.", call. = FALSE)
      }
      K <- check_constants(K)
      if (!is.null(n.burn) & !is.null(T)) {
        stopifnot(is.wholenumber(n.burn))
        T <- check_constants(T)
        if (!(n.burn < T))
          stop("Burn-in is too large; burn-in must be smaller than MCMC iterations T", call. = FALSE)
      }
      return(list(X = X, DL = DL, Z = Z, K = K, T = T))
    } #End of check_imputation

    check <- check_imputation(X = X, DL = DL, Z = Z, K = K, T = 5, n.burn = 4L, verbose)
    X <- check$X
    DL <- check$DL
    if (is.null(Z)) {
      Z <- matrix(1, nrow = nrow(X), ncol = 1, dimnames = list(NULL, "Intercept"))
    }
    else {
      l <- check_covariates(Z, X)
      X <- l$X
      Z <- l$Z
    }
    K <- check$K
    if (verbose) {
      cat("X \n")
      #print(summary(X))
      cat("DL \n")
      #print(DL)
      cat("Z \n")
      #print(summary(Z))
      cat("K =", K, "\n")
    }
    n <- nrow(X)
    c <- ncol(X)
    chemical.name <- colnames(X)
    results_Lubin <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    bootstrap_index <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    indicator.miss <- rep(NA, c)
    for (j in 1:c) {
      answer <- impute.Lubin(chemcol = X[, j], dlcol = DL[j], Z = Z, K = K)
      results_Lubin[, j, ] <- as.array(answer$imputed_values, dim = c(n, j, K))
      bootstrap_index[, j, ] <- array(answer$bootstrap_index, dim = c(n, 1, K))
      indicator.miss[j] <- answer$indicator.miss
    }
    total.miss <- sum(indicator.miss)
    if (total.miss == 0) {
      #message("#> Check: The total number of imputed values that are above the detection limit is ", sum(indicator.miss), ".")
    }
    else {
      #print(indicator.miss)
      stop("#> Error: Incorrect imputation is performed; some imputed values are above the detection limit. Could some of `X` have complete data under a lower bound (`DL`)?", call. = FALSE)
    }
    return(list(X.imputed = results_Lubin, bootstrap_index = answer$bootstrap_index, indicator.miss = total.miss))
  }
  ######################
  # End of "impute.boot"
  ######################

  fam = "gaussian" #Specifies the family for the outcomes, and is either "gaussian", "poisson", "gamma", or "binomial"
  beta_work = beta #Used in the empirical covariance weighting matrix.  It is the fixed initial working values of the parameter estimates
  obs = lapply(split(id,id),"length")
  nobs <- as.numeric(obs)             #Vector of cluster sizes for each cluster
  nsub <- length(nobs)                #The number of independent clusters or subjects

  #Substitution method using LOD
  if(substitue=="None") cen_y <- y

  #Substitution method using LOD
  if(substitue=="LOD") cen_y <- ifelse(y>=lod,y,lod)

  #Substitution method using LOD/2
  if(substitue=="LOD2") cen_y <- ifelse(y>=lod,y,lod/2)

  #Substitution method using LOD/square root of 2
  if(substitue=="LODS2") cen_y <- ifelse(y>=lod,y,lod/sqrt(2))

  #Beta-substitution method using mean
  if(substitue=="BetaMean"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_mean <- length(y)/(length(y)-length(y1))*pnorm(z-sy)*exp(-sy*z+(sy)^2/2)
    cen_y <- ifelse(y>=lod, y, lod*beta_mean)
  }
  #Beta-substitution method using geometric mean
  if(substitue=="BetaGM"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_GM <- exp((-(length(y)-(length(y)-length(y1)))*length(y))/(length(y)-length(y1))*log(f_sy_z)-sy*z-(length(y)-(length(y)-length(y1)))/(2*(length(y)-length(y1))*length(y))*(sy)^2)
    cen_y <- ifelse(y>=lod, y, lod*beta_GM)
  }
  #Multiple imputation method with one covariate using id
  if(substitue=="MIWithID"){
    y1 <- ifelse(y>=lod,y,NA)
    results_boot2 <- impute.boot(X=y1, DL=lod, Z=id, K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot2$X.imputed, 1, mean))
  }
  #Multiple imputation method with two covariates using id and visit
  if(substitue=="MIWithIDRM"){
    y1 <- ifelse(y>=lod,y,NA)
    Z2=NULL
    for(i in 1:nsub){
      Z1=as.matrix(1:nobs[i])
      Z2=rbind(Z2,Z1)
    }
    results_boot3 <- impute.boot(X=y1, DL=lod, Z=cbind(id,Z2), K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot3$X.imputed, 1, mean))
  }
  #Multiple imputation method using QQ-plot approach
  if(substitue=="QQplot"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)

    obs <- rank(y)
    rank <- (obs-0.5)/length(y)
    zscore0 <- qnorm(rank)*ifelse(y>=lod,1,0)
    zscore1 <- zscore0[which(zscore0!=0)]

    data_frame0=data.frame(cbind(lny1,zscore1))
    beta_est=as.matrix(glm(lny1~zscore1, data=data_frame0, family=gaussian)$coefficients)
    lny_zscore <- beta_est[1,1] + beta_est[2,1]*qnorm(rank)
    y_zscore <- exp(lny_zscore)
    cen_y <- ifelse(y>=lod,y,y_zscore)
  }
  y <- cen_y

  tol=0.00000001 #Used to determine convergence

  #Creating the proper design matrix
  one = matrix(1,dim(x)[1])
  x = as.matrix(cbind(one,x))
  type=as.matrix(c(1,typetd))    #Time-dependent covariate types

  np = length(beta[,1])               #Number of regression parameters

  if (corstr == "exchangeable" )
  {
    N_star = 0
    for(i in 1:nsub)
      N_star = N_star + 0.5*nobs[i]*(nobs[i]-1)  #Used in estimating the correlation
  }

  if (corstr == "AR-1" )   K_1 = sum(nobs) - nsub

  if ((corstr == "exchangeable") | (corstr == "AR-1"))
    corr_par = 0                      #This is the correlation parameter, given an initial value of 0

  scale = 0                           #Used for the correlation estimation.
  N=sum(nobs)                         #Used in estimating the scale parameter.

  iteration <- 0
  betanew <- beta

  count=1  #Used for checking that the QIF value has decreased from the previous iteration.

  #The change, or difference, in quadratic inference function values determines convergence.  maxiter is used to stop running the code if there is somehow non-convergence.

  QIFdiff=tol+1

  while (QIFdiff > tol && iteration < maxiter) {
    beta <- betanew

    arsumg <- matrix(rep(0, 2 * np), nrow = 2 * np)               #Extended score equations
    arsumc <- matrix(rep(0, 2 * np * 2 * np), nrow = 2 * np)      #Empirical weighting matrix divided by the number of independent subjects or clusters
    gi <- matrix(rep(0, 2 * np), nrow = 2 * np)                   #Extended score equation component from subject or cluster i
    gi_est <- matrix(rep(0, 2 * np), nrow = 2 * np)               #Extended score equation component from subject or cluster i
    arsumgfirstdev <- matrix(rep(0, 2 * np * np), nrow = 2 * np)  #Expected value of the derivative of arsumg
    firstdev <- matrix(rep(0, 2 * np * np), nrow = 2 * np)        #Expected value of the derivative of gi
    I_ind = matrix(0,np,np)                                       #Used for CIC: model-based covariance matrix assuming independence

    if ((corstr == "exchangeable") | (corstr == "AR-1"))
      sum = 0
    sum_scale = 0

    loc1 <- 0
    loc2 <- 0
    for(i in 1:nsub){
      loc1 <- loc2 + 1
      loc2 <- loc1 + nobs[i] - 1
      yi <- as.matrix(y[loc1:loc2, ]) #Outcomes for the ith cluster or subject
      xi <- x[loc1:loc2, ]            #Covariates for the ith cluster or subject
      ni <- nrow(yi)                  #Cluster size (or number of repeated measures)
      if(ni==1)
        xi=t(xi)

      m1 <- diag(ni)                  #First basis matrix

      #Setting up the second basis matrix
      #################################################################################
      if(corstr == "exchangeable") {
        m2 <- matrix(rep(1, ni * ni), ni) - m1

        m2_Type1 = m2

        m2_mod = m2
        m2_mod[upper.tri(m2_mod)] = 0
        m2_Type2 = m2_mod

        m2_mode = m2
        m2_mode[upper.tri(m2_mode)] = 0
        m2_mode[lower.tri(m2_mode)] = 0
        m2_Type3 = m2_mode

        m2_model = m2
        m2_model[lower.tri(m2_model)] = 0
        m2_Type4 = m2_model
      }

      if(corstr == "AR-1") {
        m2 <- matrix(rep(0, ni * ni), ni)
        for(k in 1:ni) {
          for(l in 1:ni) {
            if(abs(k-l) == 1)
              m2[k, l] <- 1
          }
        }
        m2_Type1 = m2

        m2_mod = m2
        m2_mod[upper.tri(m2_mod)] = 0
        m2_Type2 = m2_mod

        m2_mode = m2
        m2_mode[upper.tri(m2_mode)] = 0
        m2_mode[lower.tri(m2_mode)] = 0
        m2_Type3 = m2_mode

        m2_model = m2
        m2_model[lower.tri(m2_model)] = 0
        m2_Type4 = m2_model
      }
      #################################################################################

      if (fam == "gaussian") {
        ui_est = xi %*% beta       #Matrix of marginal means
        ui <- xi %*% beta_work
        fui <- ui
        fui_dev <- diag(ni)        #Diagonal matrix of marginal variances (common dispersion term is not necessary), or the derivative of the marginal mean with respect to the linear predictor
        vui <- diag(ni)            #Diagonal matrix of the inverses of the square roots of the marginal variances (common dispersion term is not necessary)
        Pearson = (yi-ui)
      }
      if (fam == "poisson") {
        ui_est <- exp(xi %*% beta)
        ui <- exp(xi %*% beta_work)
        fui <- log(ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui))
        if(ni==1)
          fui_dev <- diag(ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui)))
        if(ni==1)
          vui <- diag(sqrt(1/ui))
        Pearson = (yi-ui)/sqrt(ui)
      }
      if (fam == "gamma") {
        ui_est <- 1/(xi %*% beta)
        ui <- 1/(xi %*% beta_work)
        fui <- 1/ui
        if(ni>1)
          fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
        if(ni==1)
          fui_dev <- -diag(ui) %*% diag(ui)
        if(ni>1)
          vui <- diag(as.vector(1/ui))
        if(ni==1)
          vui <- diag(1/ui)
        Pearson = (yi-ui)/ui
      }
      if (fam == "binomial") {
        ui_est <- 1/(1 + exp(-xi %*% beta))
        ui <- 1/(1 + exp(-xi %*% beta_work))
        fui <- log(ui) - log(1 - ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
        if(ni==1)
          fui_dev <- diag(ui) %*% diag(1 - ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
        if(ni==1)
          vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
        Pearson = (yi-ui)/sqrt(ui*(1-ui))
      }

      Di = fui_dev %*% xi
      DiT = t(Di)
      wi = DiT %*% vui %*% m1 %*% vui
      I_ind = I_ind + t(xi) %*% fui_dev %*%  vui %*% vui %*% fui_dev %*% xi  #For the CIC
      zi = matrix(0,np,ni)

      #################################################################################
      for(j in 1:np) {
        if(type[j,1]==1)
          zi[j,] <- DiT[j,] %*% vui %*% m2_Type1 %*% vui
        if(type[j,1]==2)
          zi[j,] <- DiT[j,] %*% vui %*% m2_Type2 %*% vui
        if(type[j,1]==3)
          zi[j,] <- DiT[j,] %*% vui %*% m2_Type3 %*% vui
        if(type[j,1]==4)
          zi[j,] <- DiT[j,] %*% vui %*% m2_Type4 %*% vui
      }
      #################################################################################

      gi0_est <- (1/nsub) * wi %*% (yi - ui_est)
      gi1_est <- (1/nsub) * zi %*% (yi - ui_est)
      gi_est[1:np, ] <- gi0_est
      gi_est[(np + 1):(2 * np), ] <- gi1_est
      arsumg <- arsumg + gi_est                      #Creating the Extended Score vector

      gi0 <- (1/nsub) * wi %*% (yi - ui)
      gi1 <- (1/nsub) * zi %*% (yi - ui)
      gi[1:np, ] <- gi0
      gi[(np + 1):(2 * np), ] <- gi1
      arsumc <- arsumc + gi %*% t(gi)                #Creating the Empirical Covariance Matrix

      di0 <- -(1/nsub) * wi %*% fui_dev %*% xi
      di1 <- -(1/nsub) * zi %*% fui_dev %*% xi
      firstdev[1:np, ] <- di0
      firstdev[(np + 1):(2 * np), ] <- di1
      arsumgfirstdev <- arsumgfirstdev + firstdev

      #Now for obtaining the sums (from each independent subject or cluster) used in estimating the correlation parameter and also scale parameter.

      if (ni > 1) {
      for(j in 1:ni)
        sum_scale = sum_scale + Pearson[j,1]^2

      if (corstr == "exchangeable") {
        for(j in 1:(ni-1))
          for(k in (j+1):ni)
            sum = sum + Pearson[j,1]*Pearson[k,1]
      }

      if (corstr == "AR-1") {
        for(j in 1:(ni-1))
          sum = sum + Pearson[j,1]*Pearson[(j+1),1]
      }

    }
    }

    arcinv = ginv(arsumc)
    arqif1dev <- t(arsumgfirstdev) %*% arcinv %*% arsumg           #The estimating equations
    arqif2dev <- t(arsumgfirstdev) %*% arcinv %*% arsumgfirstdev   #J_N * N
    invarqif2dev <- ginv(arqif2dev)                                #The estimated asymptotic covariance matrix

    betanew <- beta - invarqif2dev %*% arqif1dev                   #Updating parameter estimates


    Q_current <- t(arsumg) %*% arcinv %*% arsumg                   #Current value of the quadratic inference function (QIF)


    if(iteration>0) QIFdiff = abs(Q_prev-Q_current)                #Convergence criterion

    bad_estimate=0      #Used to make sure the variable count has the correct value

    #Now to check that the Q decreased since the last iteration (assuming we are at least on the 2nd iteration)
    if(iteration>=1)
    {
      OK=0
      if( (Q_current>Q_prev) & (Q_prev<0) ) #If the QIF value becomes positive, or at least closer to 0, again after being negative (this is not likely), then keep these estimates.
      {
        OK=1
        bad_estimate=0
      }

      if( (Q_current>Q_prev | Q_current<0) & OK==0 )   #Checking if the QIF value has decreased. Also ensuring that the QIF is positive.  Rarely, it can be negative.
      {
        beta <- betaold - (.5^count)*invarqif2dev_old %*% arqif1dev_old
        count = count + 1
        Q_current = Q_prev
        QIFdiff=tol+1       #Make sure the code runs another iteration of the while loop
        betanew=beta
        bad_estimate=1      #used to make sure the variable count is not reset to 1
      }
    }

    if(bad_estimate==0)
    {
      betaold=beta                     #Saving current values for the previous code in the next iteration (if needed)
      invarqif2dev_old=invarqif2dev
      arqif1dev_old=arqif1dev
      count=1                          #count needs to be 1 if the QIF value is positive and has decreased
    }



    Q_prev = Q_current

    if(iteration==0) QIFdiff=tol+1 #Making sure at least two iterations are run

    iteration <- iteration + 1

  }


  #Updating the estimate of the correlation parameter

  scale=sum_scale/(N-np)

  if (corstr == "exchangeable")
    corr_par = sum/(N_star-np)/scale

  if (corstr == "AR-1")
    corr_par = sum/(K_1-np)/scale

  if (corstr == "unstructured")
    corr_par = sum/(nsub-np)/scale


  #Redefined terms:

  term = arsumgfirstdev
  emp_var = invarqif2dev
  C_N_inv = arcinv
  g_N=arsumg
  J_N = t(term) %*% C_N_inv %*% term / nsub


  #Obtaining G to correct/account for the covariance inflation (the for loop is used to obtain the derivative with respect to the empirical covariance matrix):


  firstdev <- matrix(rep(0, 2 * np * np), nrow = 2 * np)
  G=matrix(0,np,np)
  gi <- matrix(rep(0, 2 * np), nrow = 2 * np)

  loc1 <- 0
  loc2 <- 0
  for(i in 1:nsub){
    loc1 <- loc2 + 1
    loc2 <- loc1 + nobs[i] - 1
    yi <- as.matrix(y[loc1:loc2, ])
    xi <- x[loc1:loc2, ]
    ni <- nrow(yi)
    if(ni==1)
      xi=t(xi)

    m1 <- diag(ni)

    #################################################################################
    if(corstr == "exchangeable") {
      m2 <- matrix(rep(1, ni * ni), ni) - m1

      m2_Type1 = m2

      m2_mod = m2
      m2_mod[upper.tri(m2_mod)] = 0
      m2_Type2 = m2_mod

      m2_mode = m2
      m2_mode[upper.tri(m2_mode)] = 0
      m2_mode[lower.tri(m2_mode)] = 0
      m2_Type3 = m2_mode

      m2_model = m2
      m2_model[lower.tri(m2_model)] = 0
      m2_Type4 = m2_model
    }

    if(corstr == "AR-1") {
      m2 <- matrix(rep(0, ni * ni), ni)
      for(k in 1:ni) {
        for(l in 1:ni) {
          if(abs(k-l) == 1)
            m2[k, l] <- 1
        }
      }
      m2_Type1 = m2

      m2_mod = m2
      m2_mod[upper.tri(m2_mod)] = 0
      m2_Type2 = m2_mod

      m2_mode = m2
      m2_mode[upper.tri(m2_mode)] = 0
      m2_mode[lower.tri(m2_mode)] = 0
      m2_Type3 = m2_mode

      m2_model = m2
      m2_model[lower.tri(m2_model)] = 0
      m2_Type4 = m2_model
    }
    #################################################################################

    if (fam == "gaussian") {
      ui <- xi %*% beta_work
      fui <- ui
      fui_dev <- diag(ni)
      vui <- diag(ni)
    }
    if (fam == "poisson") {
      ui <- exp(xi %*% beta_work)
      fui <- log(ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui))
      if(ni==1)
        fui_dev <- diag(ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui)))
      if(ni==1)
        vui <- diag(sqrt(1/ui))
    }
    if (fam == "gamma") {
      ui <- 1/(xi %*% beta_work)
      fui <- 1/ui
      if(ni>1)
        fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
      if(ni==1)
        fui_dev <- -diag(ui) %*% diag(ui)
      if(ni>1)
        vui <- diag(as.vector(1/ui))
      if(ni==1)
        vui <- diag(1/ui)
    }
    if (fam == "binomial") {
      ui <- 1/(1 + exp(-xi %*% beta_work))
      fui <- log(ui) - log(1 - ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
      if(ni==1)
        fui_dev <- diag(ui) %*% diag(1 - ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
      if(ni==1)
        vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
    }

    Di = fui_dev %*% xi
    DiT = t(Di)
    wi = DiT %*% vui %*% m1 %*% vui
    zi = matrix(0,np,ni)

    #################################################################################
    for(j in 1:np) {
      if(type[j,1]==1)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type1 %*% vui
      if(type[j,1]==2)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type2 %*% vui
      if(type[j,1]==3)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type3 %*% vui
      if(type[j,1]==4)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type4 %*% vui
    }
    #################################################################################

    gi[1:np, ]=as.matrix(wi %*% (yi - ui))
    gi[(np + 1):(2 * np), ]=as.matrix(zi %*% (yi - ui))

    di0 <- - wi %*% fui_dev %*% xi
    di1 <- - zi %*% fui_dev %*% xi
    firstdev[1:np, ] <- di0
    firstdev[(np + 1):(2 * np), ] <- di1

    for(j in 1:np)
      G[,j] = G[,j] + ( ginv(J_N)%*%t(term)%*%(C_N_inv/nsub)%*%( firstdev[,j]%*%t(gi) + gi%*%t(firstdev[,j]) )%*%(C_N_inv/nsub)%*%g_N )/nsub
  }



  #Correcting for the other source of bias in the estimated asymptotic covariance matrix :

  arsumc_BC_1 <- matrix(rep(0, 2 * np * 2 * np), nrow = 2 * np)
  arsumc_BC_2 <- matrix(rep(0, 2 * np * 2 * np), nrow = 2 * np)
  gi_BC_1 <- matrix(rep(0, 2 * np), nrow = 2 * np)
  gi_BC_2 <- matrix(rep(0, 2 * np), nrow = 2 * np)

  loc1 <- 0
  loc2 <- 0
  for(i in 1:nsub){
    loc1 <- loc2 + 1
    loc2 <- loc1 + nobs[i] - 1
    yi <- as.matrix(y[loc1:loc2, ])
    xi <- x[loc1:loc2, ]
    ni <- nrow(yi)
    if(ni==1)
      xi=t(xi)

    m1 <- diag(ni)

    #################################################################################
    if(corstr == "exchangeable") {
      m2 <- matrix(rep(1, ni * ni), ni) - m1

      m2_Type1 = m2

      m2_mod = m2
      m2_mod[upper.tri(m2_mod)] = 0
      m2_Type2 = m2_mod

      m2_mode = m2
      m2_mode[upper.tri(m2_mode)] = 0
      m2_mode[lower.tri(m2_mode)] = 0
      m2_Type3 = m2_mode

      m2_model = m2
      m2_model[lower.tri(m2_model)] = 0
      m2_Type4 = m2_model
    }

    if(corstr == "AR-1") {
      m2 <- matrix(rep(0, ni * ni), ni)
      for(k in 1:ni) {
        for(l in 1:ni) {
          if(abs(k-l) == 1)
            m2[k, l] <- 1
        }
      }

      m2_Type1 = m2

      m2_mod = m2
      m2_mod[upper.tri(m2_mod)] = 0
      m2_Type2 = m2_mod

      m2_mode = m2
      m2_mode[upper.tri(m2_mode)] = 0
      m2_mode[lower.tri(m2_mode)] = 0
      m2_Type3 = m2_mode

      m2_model = m2
      m2_model[lower.tri(m2_model)] = 0
      m2_Type4 = m2_model
    }
    #################################################################################

    if (fam == "gaussian") {
      ui <- xi %*% beta
      fui <- ui
      fui_dev <- diag(ni)
      vui <- diag(ni)
    }
    if (fam == "poisson") {
      ui <- exp(xi %*% beta)
      fui <- log(ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui))
      if(ni==1)
        fui_dev <- diag(ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui)))
      if(ni==1)
        vui <- diag(sqrt(1/ui))
    }
    if (fam == "gamma") {
      ui <- 1/(xi %*% beta)
      fui <- 1/ui
      if(ni>1)
        fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
      if(ni==1)
        fui_dev <- -diag(ui) %*% diag(ui)
      if(ni>1)
        vui <- diag(as.vector(1/ui))
      if(ni==1)
        vui <- diag(1/ui)
    }
    if (fam == "binomial") {
      ui <- 1/(1 + exp(-xi %*% beta))
      fui <- log(ui) - log(1 - ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
      if(ni==1)
        fui_dev <- diag(ui) %*% diag(1 - ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
      if(ni==1)
        vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
    }

    Di = fui_dev %*% xi
    DiT = t(Di)
    wi = DiT %*% vui %*% m1 %*% vui
    zi = matrix(0,np,ni)

    #################################################################################
    for(j in 1:np) {
      if(type[j,1]==1)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type1 %*% vui
      if(type[j,1]==2)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type2 %*% vui
      if(type[j,1]==3)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type3 %*% vui
      if(type[j,1]==4)
        zi[j,] <- DiT[j,] %*% vui %*% m2_Type4 %*% vui
    }
    #################################################################################


    #Correction similar to the one proposed by Mancl and DeRouen (2001) with GEE:
    #O_i = Hi%*%Ri
    Ident = diag(ni)
    Hi = fui_dev %*% xi %*% (diag(np)+G) %*% emp_var %*% t(term) %*% C_N_inv / nsub
    Mi = matrix(0,ni,np)

    #################################################################################
    for(j in 1:np) {
      if(type[j,1]==1)
        Mi[,j] = vui %*% m2_Type1 %*% vui %*% Di[,j]
      if(type[j,1]==2)
        Mi[,j] = vui %*% m2_Type2 %*% vui %*% Di[,j]
      if(type[j,1]==3)
        Mi[,j] = vui %*% m2_Type3 %*% vui %*% Di[,j]
      if(type[j,1]==4)
        Mi[,j] = vui %*% m2_Type4 %*% vui %*% Di[,j]
    }
    Ri = rbind(t(vui %*% m1 %*% vui %*% Di), t(Mi))

    #################################################################################

    #Replacing estimated empirical covariances with their bias corrected versions
    gi0_BC_1 <- (1/nsub) * wi %*% ginv(Ident+Hi%*%Ri) %*% (yi - ui)
    gi1_BC_1 <- (1/nsub) * zi %*% ginv(Ident+Hi%*%Ri) %*% (yi - ui)
    gi0_BC2_1 <- (1/nsub) * t((yi - ui)) %*% ginv(Ident+t(Hi%*%Ri)) %*% t(wi)
    gi1_BC2_1 <- (1/nsub) * t((yi - ui)) %*% ginv(Ident+t(Hi%*%Ri)) %*% t(zi)

    #Obtaining the central bias-corrected empirical covariance matrix (but divided by the number of subjects or clusters) inside our proposed covariance formula
    gi_BC_1[1:np, ] <- gi0_BC_1
    gi_BC_1[(np + 1):(2 * np), ] <- gi1_BC_1
    gi_BC2_1=matrix(0,1,2*np)
    gi_BC2_1[,1:np] <- gi0_BC2_1
    gi_BC2_1[,(np + 1):(2 * np)] <-gi1_BC2_1
    arsumc_BC_1 <- arsumc_BC_1 + gi_BC_1 %*% gi_BC2_1

    #Correction similar to the one proposed by Kauermann and Carroll (2001) with GEE:
    gi0_BC_2 <- (1/nsub) * wi %*% ginv(Ident+Hi%*%Ri) %*% (yi - ui)
    gi1_BC_2 <- (1/nsub) * zi %*% ginv(Ident+Hi%*%Ri) %*% (yi - ui)
    gi0_BC2_2 <- (1/nsub) * t((yi - ui)) %*% t(wi)
    gi1_BC2_2 <- (1/nsub) * t((yi - ui)) %*% t(zi)

    gi_BC_2[1:np, ] <- gi0_BC_2
    gi_BC_2[(np + 1):(2 * np), ] <- gi1_BC_2
    gi_BC2_2=matrix(0,1,2*np)
    gi_BC2_2[,1:np] <- gi0_BC2_2
    gi_BC2_2[,(np + 1):(2 * np)] <-gi1_BC2_2
    arsumc_BC_2 <- arsumc_BC_2 + gi_BC_2 %*% gi_BC2_2

  } #The end of the for loop, iterating through all nsub subjects


  #Here are the bias-corrected covariance estimates:
  covariance_MD = (diag(np)+G) %*% emp_var %*% t(term) %*% C_N_inv %*% arsumc_BC_1 %*% C_N_inv %*% term %*% emp_var %*% t(diag(np)+G)
  covariance_KC = (diag(np)+G) %*% emp_var %*% t(term) %*% C_N_inv %*% arsumc_BC_2 %*% C_N_inv %*% term %*% emp_var %*% t(diag(np)+G)
  covariance_AVG = (covariance_MD + covariance_KC)/2

  I_ind = I_ind/scale

  beta_number=as.matrix(seq(1:np))-1 #Used to indicate the parameter. For instance, 0 denotes the intercept.
  SE_A=SE_MD=SE_KC=SE_AVG=TECM_A=TECM_MD=TECM_KC=TECM_AVG=CIC_A=CIC_MD=CIC_KC=CIC_AVG=matrix(0,np,1)
  for(j in 1:np){
    SE_A[j,1]=sqrt(emp_var[j,j])
    SE_MD[j,1]=sqrt(covariance_MD[j,j])
    SE_KC[j,1]=sqrt(covariance_KC[j,j])
    SE_AVG[j,1]=sqrt(covariance_AVG[j,j])}
  Wald_A=Wald_MD=Wald_KC=Wald_AVG=matrix(0,np,1)
  for(j in 1:np){
    Wald_A[j,1]=beta[j,1]/SE_A[j,1]
    Wald_MD[j,1]=beta[j,1]/SE_MD[j,1]
    Wald_KC[j,1]=beta[j,1]/SE_KC[j,1]
    Wald_AVG[j,1]=beta[j,1]/SE_AVG[j,1]}
  df=nsub-np
  p_valueA=p_valueMD=p_valueKC=p_valueAVG=matrix(0,np,1)
  for(j in 1:np){
    p_valueA[j,1]=1-pf((Wald_A[j,1])^2,1,df)
    p_valueMD[j,1]=1-pf((Wald_MD[j,1])^2,1,df)
    p_valueKC[j,1]=1-pf((Wald_KC[j,1])^2,1,df)
    p_valueAVG[j,1]=1-pf((Wald_AVG[j,1])^2,1,df)}

  TECM_A[1,1]=sum(diag( emp_var ))
  TECM_MD[1,1]=sum(diag( covariance_MD ))
  TECM_KC[1,1]=sum(diag( covariance_KC ))
  TECM_AVG[1,1]=sum(diag( covariance_KC ))
  CIC_A[1,1]=sum(diag( I_ind %*% emp_var ))
  CIC_MD[1,1]=sum(diag( I_ind %*% covariance_MD ))
  CIC_KC[1,1]=sum(diag( I_ind %*% covariance_KC ))
  CIC_AVG[1,1]=sum(diag( I_ind %*% covariance_AVG ))

  output_A=cbind(beta_number,beta,SE_A,Wald_A,p_valueA,TECM_A,CIC_A)
  output_MD=cbind(beta_number,beta,SE_MD,Wald_MD,p_valueMD,TECM_MD,CIC_MD)
  output_KC=cbind(beta_number,beta,SE_KC,Wald_KC,p_valueKC,TECM_KC,CIC_KC)
  output_AVG=cbind(beta_number,beta,SE_AVG,Wald_AVG,p_valueAVG,TECM_AVG,CIC_AVG)

  colnames(output_A) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_MD) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_KC) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_AVG) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")

  correction_A=correction_MD=correction_KC=correction_AVG=matrix(0,1,7)
  correction_A[1,]=c("Typical Asymptotic SE","","","","","","")
  correction_MD[1,]=c("Bias-Corrected SE (MD)","","","","","","")
  correction_KC[1,]=c("Bias-Corrected SE (KC)","","","","","","")
  correction_AVG[1,]=c("Bias-Corrected SE (AVG)","","","","","","")
  output_ALL=rbind(correction_A,output_A,correction_MD,output_MD,correction_KC,output_KC,correction_AVG,output_AVG)

  message("Results include typical asymptotic standard error (SE) estimates and bias-corrected SE estimates that use covariance inflation corrections.")
  message("Covariance inflation corrections are based on the corrections of Mancl and DeRouen (MD) (2001), and Kauermann and Carroll (KC) (2001).")
  message("Last covariance inflation correction averages the above corrections (AVG) (Ford and Westgate, 2017, 2018).")
  return(knitr::kable(output_ALL))
}










#Function for modified GMM

Modified.GMM=function(id, y, x, lod, substitue, beta, maxiter)
{
  #############################################################
  # "impute.boot" function from "miWQS" package (version 0.4.4)
  #############################################################
  impute.boot=function(X, DL, Z = NULL, K = 5L, verbose = FALSE)
  {
    check_function.Lub=function(chemcol, dlcol, Z, K, verbose)
    {
      X <- check_X(as.matrix(chemcol))
      if (ncol(X) > 1) {
        warning("This approach cannot impute more than one chemical at time. Imputing first element...")
        chemcol <- chemcol[1]
      }
      else {
        chemcol <- as.vector(X)
      }
      if (is.null(dlcol))
        stop("dlcol is NULL. A detection limit is needed", call. = FALSE)
      if (is.na(dlcol))
        stop("The detection limit has missing values so chemical is not imputed.",
             immediate. = FALSE, call. = FALSE)
      if (!is.numeric(dlcol))
        stop("The detection limit is non-numeric. Detection limit must be numeric.",
             call. = FALSE)
      if (!is.vector(dlcol))
        stop("The detection limit must be a vector.", call. = FALSE)
      if (length(dlcol) > 1) {
        if (min(dlcol, na.rm = TRUE) == max(dlcol, na.rm = TRUE)) {
          dlcol <- unique(dlcol)
        }
        else {
          warning(" The detection limit is not unique, ranging from ",
                  min(dlcol, na.rm = TRUE), " to ", max(dlcol,
                                                        na.rm = TRUE), "; The smallest value is assumed to be detection limit",
                  call. = FALSE, immediate. = FALSE)
          detcol <- !is.na(chemcol)
          if (verbose) {
            cat("\n Summary when chemical is missing (BDL) \n")
            #print(summary(dlcol[detcol == 0]))
            cat("## when chemical is observed \n ")
            #print(summary(dlcol[detcol == 1]))
            dlcol <- min(dlcol, na.rm = TRUE)
          }
        }
      }
      if (is.null(Z)) {
        Z <- matrix(rep(1, length(chemcol)), ncol = 1)
      }
      if (anyNA(Z)) {
        warning("Missing covariates are ignored.")
      }
      if (!is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      K <- check_constants(K)
      return(list(chemcol = chemcol, dlcol = dlcol, Z = Z, K = K))
    } #End of check_function.Lub

    impute.Lubin=function(chemcol, dlcol, Z = NULL, K = 5L, verbose = FALSE)
    {
      l <- check_function.Lub(chemcol, dlcol, Z, K, verbose)
      dlcol <- l$dlcol
      Z <- l$Z
      K <- l$K
      n <- length(chemcol)
      chemcol2 <- ifelse(is.na(chemcol), 1e-05, chemcol)
      event <- ifelse(is.na(chemcol), 3, 1)
      fullchem_data <- na.omit(data.frame(id = seq(1:n), chemcol2 = chemcol2,
                                          event = event, LB = rep(0, n), UB = rep(dlcol, n), Z))
      n <- nrow(fullchem_data)
      if (verbose) {
        cat("Dataset Used in Survival Model \n")
        #print(head(fullchem_data))
      }
      bootstrap_data <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      beta_not <- NA
      std <- rep(0, K)
      unif_lower <- rep(0, K)
      unif_upper <- rep(0, K)
      imputed_values <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      for (a in 1:K) {
        bootstrap_data[, a] <- as.vector(sample(1:n, replace = TRUE))
        data_sample <- fullchem_data[bootstrap_data[, a], ]
        freqs <- as.data.frame(table(bootstrap_data[, a]))
        freqs_ids <- as.numeric(as.character(freqs[, 1]))
        my.weights <- freqs[, 2]
        final <- fullchem_data[freqs_ids, ]
        my.surv.object <- survival::Surv(time = final$chemcol2,
                                         time2 = final$UB, event = final$event, type = "interval")
        model <- survival::survreg(my.surv.object ~ ., data = final[,
                                                                    -(1:5), drop = FALSE], weights = my.weights, dist = "lognormal",
                                   x = TRUE)
        beta_not <- rbind(beta_not, model$coefficients)
        std[a] <- model$scale
        Z <- model$x
        mu <- Z %*% beta_not[a + 1, ]
        unif_lower <- plnorm(1e-05, mu, std[a])
        unif_upper <- plnorm(dlcol, mu, std[a])
        u <- runif(n, unif_lower, unif_upper)
        imputed <- qlnorm(u, mu, std[a])
        imputed_values[, a] <- ifelse(fullchem_data$chemcol2 ==
                                        1e-05, imputed, chemcol)
      }
      x.miss.index <- ifelse(chemcol == 0 | is.na(chemcol), TRUE,
                             FALSE)
      indicator.miss <- sum(imputed_values[x.miss.index, ] > dlcol)
      if (verbose) {
        beta_not <- beta_not[-1, , drop = FALSE]
        cat("\n ## MLE Estimates \n")
        A <- round(cbind(beta_not, std), digits = 4)
        colnames(A) <- c(names(model$coefficients), "stdev")
        #print(A)
        cat("## Uniform Imputation Range \n")
        B <- rbind(format(range(unif_lower), digits = 3, scientific = TRUE),
                   format(range(unif_upper), digits = 3, nsmall = 3))
        rownames(B) <- c("unif_lower", "unif_upper")
        #print(B)
        cat("## Detection Limit:", unique(dlcol), "\n")
      }
      bootstrap_data <- apply(bootstrap_data, 2, as.factor)
      return(list(imputed_values = imputed_values, bootstrap_index = bootstrap_data,
                  indicator.miss = indicator.miss))
    } #End of impute.Lubin

    check_covariates=function(Z, X)
    {
      nZ <- if (is.vector(Z) | is.factor(Z)) {
        length(Z)
      }
      else {
        nrow(Z)
      }
      if (nrow(X) != nZ) {
        if (verbose) {
          cat("> X has", nrow(X), "individuals, but Z has", nZ,
              "individuals")
          stop("Can't Run. The total number of individuals with components X is different than total number of individuals with covariates Z.",
               call. = FALSE)
        }
      }
      if (anyNA(Z)) {
        p <- if (is.vector(Z) | is.factor(Z)) {
          1
        }
        else {
          ncol(Z)
        }
        index <- complete.cases(Z)
        Z <- Z[index, , drop = FALSE]
        X <- X[index, , drop = FALSE]
        dimnames(X)[[1]] <- 1:nrow(X)
        warning("Covariates are missing for individuals ", paste(which(!index),
                                                                 collapse = ", "), " and are ignored. Subjects renumbered in complete data.",
                call. = FALSE, immediate. = TRUE)
      }
      if (!is.null(Z) & !is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else if (is.array(Z)) {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      return(list(Z = Z, X = X))
    } #End of check_covariates

    is.wholenumber=function(x)
    {
      x > -1 && is.integer(x)
    } #End of is.wholenumber
    #is.wholenumber=function(x, tol = .Machine$double.eps^0.5)
    #{
    #  x > -1 && is.integer(x, tol)
    #} #End of is.wholenumber

    check_constants=function(K)
    {
      if (is.null(K))
        stop(sprintf(" must be non-null."), call. = TRUE)
      if (!is.numeric(K) | length(K) != 1)
        stop(sprintf(" must be numeric of length 1"), call. = TRUE)
      if (K <= 0)
        stop(sprintf(" must be a positive integer"), call. = TRUE)
      if (!is.wholenumber(K)) {
        K <- ceiling(K)
      }
      return(K)
    } #End of check_constants

    check_X=function(X)
    {
      if (is.null(X)) {
        stop("X is null. X must have some data.", call. = FALSE)
      }
      if (is.data.frame(X)) {
        X <- as.matrix(X)
      }
      if (!all(apply(X, 2, is.numeric))) {
        stop("X must be numeric. Some columns in X are not numeric.", call. = FALSE)
      }
      return(X)
    } #End of check_X

    check_imputation=function(X, DL, Z, K, T, n.burn, verbose = NULL)
    {
      if (is.vector(X)) {
        X <- as.matrix(X)
      }
      X <- check_X(X)
      if (!anyNA(X))
        stop("Matrix X has nothing to impute. No need to impute.", call. = FALSE)
      if (is.matrix(DL) | is.data.frame(DL)) {
        if (ncol(DL) == 1 | nrow(DL) == 1) {
          DL <- as.numeric(DL)
        }
        else {
          stop("The detection limit must be a vector.", call. = FALSE)
        }
      }
      stopifnot(is.numeric(DL))
      if (anyNA(DL)) {
        no.DL <- which(is.na(DL))
        warning("The detection limit for ", names(no.DL), " is missing.\n Both the chemical and detection limit is removed in imputation.",
                call. = FALSE)
        DL <- DL[-no.DL]
        if (!(names(no.DL) %in% colnames(X))) {
          if (verbose) {
            cat("  ##Missing DL does not match colnames(X) \n")
          }
        }
        X <- X[, -no.DL]
      }
      if (length(DL) != ncol(X)) {
        if (verbose) {
          cat("The following components \n")
        }
        X[1:3, ]
        if (verbose) {
          cat("have these detection limits \n")
        }
        DL
        stop("Each component must have its own detection limit.", call. = FALSE)
      }
      K <- check_constants(K)
      if (!is.null(n.burn) & !is.null(T)) {
        stopifnot(is.wholenumber(n.burn))
        T <- check_constants(T)
        if (!(n.burn < T))
          stop("Burn-in is too large; burn-in must be smaller than MCMC iterations T", call. = FALSE)
      }
      return(list(X = X, DL = DL, Z = Z, K = K, T = T))
    } #End of check_imputation

    check <- check_imputation(X = X, DL = DL, Z = Z, K = K, T = 5, n.burn = 4L, verbose)
    X <- check$X
    DL <- check$DL
    if (is.null(Z)) {
      Z <- matrix(1, nrow = nrow(X), ncol = 1, dimnames = list(NULL, "Intercept"))
    }
    else {
      l <- check_covariates(Z, X)
      X <- l$X
      Z <- l$Z
    }
    K <- check$K
    if (verbose) {
      cat("X \n")
      #print(summary(X))
      cat("DL \n")
      #print(DL)
      cat("Z \n")
      #print(summary(Z))
      cat("K =", K, "\n")
    }
    n <- nrow(X)
    c <- ncol(X)
    chemical.name <- colnames(X)
    results_Lubin <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    bootstrap_index <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    indicator.miss <- rep(NA, c)
    for (j in 1:c) {
      answer <- impute.Lubin(chemcol = X[, j], dlcol = DL[j], Z = Z, K = K)
      results_Lubin[, j, ] <- as.array(answer$imputed_values, dim = c(n, j, K))
      bootstrap_index[, j, ] <- array(answer$bootstrap_index, dim = c(n, 1, K))
      indicator.miss[j] <- answer$indicator.miss
    }
    total.miss <- sum(indicator.miss)
    if (total.miss == 0) {
      #message("#> Check: The total number of imputed values that are above the detection limit is ", sum(indicator.miss), ".")
    }
    else {
      #print(indicator.miss)
      stop("#> Error: Incorrect imputation is performed; some imputed values are above the detection limit. Could some of `X` have complete data under a lower bound (`DL`)?", call. = FALSE)
    }
    return(list(X.imputed = results_Lubin, bootstrap_index = answer$bootstrap_index, indicator.miss = total.miss))
  }
  ######################
  # End of "impute.boot"
  ######################

  fam = "gaussian" #Family for the outcomes
  beta_work = beta #Used in the empirical covariance weighting matrix.  It is the fixed initial working values of the parameter estimates
  obs=lapply(split(id,id),"length")
  nobs <- as.numeric(obs)             #Vector of cluster sizes for each cluster
  nsub <- length(nobs)                #The number of independent clusters or subjects

  #Substitution method using LOD
  if(substitue=="None") cen_y <- y

  #Substitution method using LOD
  if(substitue=="LOD") cen_y <- ifelse(y>=lod,y,lod)

  #Substitution method using LOD/2
  if(substitue=="LOD2") cen_y <- ifelse(y>=lod,y,lod/2)

  #Substitution method using LOD/square root of 2
  if(substitue=="LODS2") cen_y <- ifelse(y>=lod,y,lod/sqrt(2))

  #Beta-substitution method using mean
  if(substitue=="BetaMean"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_mean <- length(y)/(length(y)-length(y1))*pnorm(z-sy)*exp(-sy*z+(sy)^2/2)
    cen_y <- ifelse(y>=lod, y, lod*beta_mean)
  }
  #Beta-substitution method using geometric mean
  if(substitue=="BetaGM"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_GM <- exp((-(length(y)-(length(y)-length(y1)))*length(y))/(length(y)-length(y1))*log(f_sy_z)-sy*z-(length(y)-(length(y)-length(y1)))/(2*(length(y)-length(y1))*length(y))*(sy)^2)
    cen_y <- ifelse(y>=lod, y, lod*beta_GM)
  }
  #Multiple imputation method with one covariate using id
  if(substitue=="MIWithID"){
    y1 <- ifelse(y>=lod,y,NA)
    results_boot2 <- impute.boot(X=y1, DL=lod, Z=id, K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot2$X.imputed, 1, mean))
  }
  #Multiple imputation method with two covariates using id and visit
  if(substitue=="MIWithIDRM"){
    y1 <- ifelse(y>=lod,y,NA)
    Z2=NULL
    for(i in 1:nsub){
      Z1=as.matrix(1:nobs[i])
      Z2=rbind(Z2,Z1)
    }
    results_boot3 <- impute.boot(X=y1, DL=lod, Z=cbind(id,Z2), K=5, verbose = TRUE)
    cen_y <- as.matrix(apply(results_boot3$X.imputed, 1, mean))
  }
  #Multiple imputation method using QQ-plot approach
  if(substitue=="QQplot"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)

    obs <- rank(y)
    rank <- (obs-0.5)/length(y)
    zscore0 <- qnorm(rank)*ifelse(y>=lod,1,0)
    zscore1 <- zscore0[which(zscore0!=0)]

    data_frame0=data.frame(cbind(lny1,zscore1))
    beta_est=as.matrix(glm(lny1~zscore1, data=data_frame0, family=gaussian)$coefficients)
    lny_zscore <- beta_est[1,1] + beta_est[2,1]*qnorm(rank)
    y_zscore <- exp(lny_zscore)
    cen_y <- ifelse(y>=lod,y,y_zscore)
  }
  y <- cen_y

  tol=0.00000001 #Used to determine convergence

  #Creating the proper design matrix
  one=matrix(1,dim(x)[1])
  x=as.matrix(cbind(one,x))

  type=as.matrix(c(3,rep(1,dim(x)[2]-1)))    #Time-dependent covariate types
  np = length(beta[,1])               #Number of regression parameters

  scale = 0                           #Used for the correlation estimation.
  N=sum(nobs)                         #Used in estimating the scale parameter.

  iteration <- 0
  betanew <- beta

  count=1  #Used for checking that the QIF value has decreased from the previous iteration.

  #The change, or difference, in quadratic inference function values determines convergence.  maxiter is used to stop running the code if there is somehow non-convergence.

  QIFdiff=tol+1

  while (QIFdiff > tol && iteration < maxiter) {
    beta <- betanew

    max_length = np*(max(nobs))^2
    arsumg1 <- matrix(0, max_length, 1)
    arsumc1 <- matrix(0, max_length, max_length)
    arsumgfirstdev1 <- matrix(0, max_length, np)
    I_ind = matrix(0, np, np)    #Used for CIC: model-based covariance matrix assuming independence
    sum_scale = 0

    loc1 <- 0
    loc2 <- 0
    for(i in 1:nsub){
      loc1 <- loc2 + 1
      loc2 <- loc1 + nobs[i] - 1
      yi <- as.matrix(y[loc1:loc2, ]) #Outcomes for the ith cluster or subject
      xi <- x[loc1:loc2, ]            #Covariates for the ith cluster or subject
      ni <- nrow(yi)                  #Cluster size (or number of repeated measures)

      if (fam == "gaussian") {
        ui_est = xi %*% beta       #Matrix of margiZEROl means
        ui <- xi %*% beta_work
        fui <- ui
        fui_dev <- diag(ni)        #Diagonal matrix of marginal variances (common dispersion term is not necessary), or the derivative of the marginal mean with respect to the linear predictor
        vui <- diag(ni)            #Diagonal matrix of the inverses of the square roots of the marginal variances (common dispersion term is not necessary)
        Pearson = (yi-ui)
      }
      if (fam == "poisson") {
        ui_est <- exp(xi %*% beta)
        ui <- exp(xi %*% beta_work)
        fui <- log(ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui))
        if(ni==1)
          fui_dev <- diag(ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui)))
        if(ni==1)
          vui <- diag(sqrt(1/ui))
        Pearson = (yi-ui)/sqrt(ui)
      }
      if (fam == "gamma") {
        ui_est <- 1/(xi %*% beta)
        ui <- 1/(xi %*% beta_work)
        fui <- 1/ui
        if(ni>1)
          fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
        if(ni==1)
          fui_dev <- -diag(ui) %*% diag(ui)
        if(ni>1)
          vui <- diag(as.vector(1/ui))
        if(ni==1)
          vui <- diag(1/ui)
        Pearson = (yi-ui)/ui
      }
      if (fam == "binomial") {
        ui_est <- 1/(1 + exp(-xi %*% beta))
        ui <- 1/(1 + exp(-xi %*% beta_work))
        fui <- log(ui) - log(1 - ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
        if(ni==1)
          fui_dev <- diag(ui) %*% diag(1 - ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
        if(ni==1)
          vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
        Pearson = (yi-ui)/sqrt(ui*(1-ui))
      }

      gi <- matrix(0, ni * ni, np)
      gi_est <- matrix(0, ni * ni, np)

      Di = fui_dev %*% xi
      DiT = t(Di)
      I_ind = I_ind + t(xi) %*% fui_dev %*%  vui %*% vui %*% fui_dev %*% xi  #For the CIC

      for(j in 1:np)
      {
        if(type[j,1]==1)
        {
          g_i_est <- (1/nsub) * Di[,j] %*% t(yi - ui_est)
          gi_est[,j] <- matrix(c(g_i_est))

          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          gi[,j] <- matrix(c(g_i))
        }
        if(type[j,1]==2)
        {
          g_i_est <- (1/nsub) * Di[,j] %*% t(yi - ui_est)
          g_i_est[upper.tri(g_i_est)] <- NA
          gi_est[,j] <- matrix(c(g_i_est))

          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          g_i[upper.tri(g_i)] <- NA
          gi[,j] <- matrix(c(g_i))
        }
        if(type[j,1]==3)
        {
          g_i_est <- (1/nsub) * Di[,j] %*% t(yi - ui_est)
          g_i_est[lower.tri(g_i_est)] <- NA
          g_i_est[upper.tri(g_i_est)] <- NA
          gi_est[,j] <- matrix(c(g_i_est))

          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          g_i[lower.tri(g_i)] <- NA
          g_i[upper.tri(g_i)] <- NA
          gi[,j] <- matrix(c(g_i))
        }
      }
      gi_est1 <- matrix(c(gi_est))
      gi_est2 <- matrix(gi_est1[which(gi_est1!="NA")])

      gi1 <- matrix(c(gi))
      gi2 <- matrix(gi1[which(gi1!="NA")])

      arsumg1 <- arsumg1 + gi_est1
      arsumg <- matrix(arsumg1[which(arsumg1!="NA")])

      arsumc1 <- arsumc1 + gi1 %*% t(gi1)
      arsumc <- matrix(arsumc1[which(arsumc1!="NA")], length(gi2), length(gi2))

      d_i=list();i=1
      di=list();i=1
      for(j in 1:np){
        for(k in 1:np){
          d_i[[i]] <- -(1/nsub) * Di[,j] %*% t(t(Di)[k,])
          di[[i]] <- matrix(c(d_i[[i]]))
          i=i+1
        }
      }

      di1 <- matrix(do.call(cbind, di), ni * ni * np, np)
      for(l in 1:(ni*ni*np)){
        if (is.na(gi1[l,1]))
          di1[l,] <- NA
      }

      firstdev1 <- di1
      arsumgfirstdev1 <- arsumgfirstdev1 + firstdev1
      arsumgfirstdev <- matrix(arsumgfirstdev1[which(arsumgfirstdev1!="NA")], length(gi2), np)

      #di2 <- matrix(0, length(gi2), np)
      #for(j in 1:np){
      #  di2[,j] <- matrix(c(di1[,j])[which(c(di1[,j])!="NA")])
      #}

      #Now for obtaining the sums (from each independent subject or cluster) used in estimating the correlation parameter and also scale parameter.
      for(j in 1:ni){
        sum_scale = sum_scale + Pearson[j,1]^2
      }
    }#The end of the for loop, iterating through all nsub subjects

    ####################################################################
    #Here is C_N and the model-based weight matrix:
    S_Reg = nsub*arsumc
    S_Proposed = nsub*(sum(diag( arsumc ))/length(gi2))*diag(length(gi2))

    #Here is d^2_N
    d_N = sum(diag( (S_Reg-S_Proposed)%*%t(S_Reg-S_Proposed) ))/(length(gi2))

    #Obtaining t^2_N
    t_N = 0
    loc1 <- 0
    loc2 <- 0
    for(i in 1:nsub){
      loc1 <- loc2 + 1
      loc2 <- loc1 + nobs[i] - 1
      yi <- as.matrix(y[loc1:loc2, ])
      xi <- x[loc1:loc2, ]
      ni <- nrow(yi)

      if (fam == "gaussian") {
        ui_est = xi %*% beta       #Matrix of marginal means
        ui <- xi %*% beta_work
        fui <- ui
        fui_dev <- diag(ni)        #Diagonal matrix of marginal variances (common dispersion term is not necessary), or the derivative of the marginal mean with respect to the linear predictor
        vui <- diag(ni)            #Diagonal matrix of the inverses of the square roots of the marginal variances (common dispersion term is not necessary)
      }
      if (fam == "poisson") {
        ui_est <- exp(xi %*% beta)
        ui <- exp(xi %*% beta_work)
        fui <- log(ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui))
        if(ni==1)
          fui_dev <- diag(ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui)))
        if(ni==1)
          vui <- diag(sqrt(1/ui))
      }
      if (fam == "gamma") {
        ui_est <- 1/(xi %*% beta)
        ui <- 1/(xi %*% beta_work)
        fui <- 1/ui
        if(ni>1)
          fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
        if(ni==1)
          fui_dev <- -diag(ui) %*% diag(ui)
        if(ni>1)
          vui <- diag(as.vector(1/ui))
        if(ni==1)
          vui <- diag(1/ui)
      }
      if (fam == "binomial") {
        ui_est <- 1/(1 + exp(-xi %*% beta))
        ui <- 1/(1 + exp(-xi %*% beta_work))
        fui <- log(ui) - log(1 - ui)
        if(ni>1)
          fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
        if(ni==1)
          fui_dev <- diag(ui) %*% diag(1 - ui)
        if(ni>1)
          vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
        if(ni==1)
          vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
      }

      gi <- matrix(0, ni * ni, np)
      gi_est <- matrix(0, ni * ni, np)

      Di = fui_dev %*% xi
      DiT = t(Di)

      for(j in 1:np)
      {
        if(type[j,1]==1)
        {
          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          gi[,j] <- matrix(c(g_i))
        }
        if(type[j,1]==2)
        {
          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          g_i[upper.tri(g_i)] <- NA
          gi[,j] <- matrix(c(g_i))
        }
        if(type[j,1]==3)
        {
          g_i <- (1/nsub) * Di[,j] %*% t(yi - ui)
          g_i[lower.tri(g_i)] <- NA
          g_i[upper.tri(g_i)] <- NA
          gi[,j] <- matrix(c(g_i))
        }
      }
      gi1 <- matrix(c(gi))
      gi2 <- matrix(gi1[which(gi1!="NA")])

      #Here I am obtaining t^2_N
      reg_res = nsub * gi2
      #reg_res1 = wi%*%(yi - ui)
      #reg_res2 = zi%*%(yi - ui)
      #reg_res = rbind(reg_res1,reg_res2)
      reg_res = reg_res%*%t(reg_res)
      t_N = t_N + sum(diag( (reg_res-S_Reg)%*%t(reg_res-S_Reg) ))/(length(gi2))

    } #The end of the for loop, iterating through all nsub subjects


    #Now to make sure t_N is not negative
    if(t_N<0)
      t_N=0

    #Now to obtain the estimated weight, rho_N
    rho_N = min(t_N,d_N)/d_N

    #Now to obtain the working weight matrix, which is the linear combination of the model-based and regular weight matrices
    # Also, divide by nsub since that is what is needed for the estimating equation
    C_N_working = ( rho_N*S_Proposed + (1-rho_N)*S_Reg )/nsub


    arcinv=ginv(C_N_working)
    ####################################################################

    #arcinv = ginv(arsumc)
    arqif1dev <- t(arsumgfirstdev) %*% arcinv %*% arsumg           #The estimating equations (U_N)
    arqif2dev <- t(arsumgfirstdev) %*% arcinv %*% arsumgfirstdev   #J_N * N
    invarqif2dev <- ginv(arqif2dev)                                #The estimated asymptotic covariance matrix

    betanew <- beta - invarqif2dev %*% arqif1dev                   #Updating parameter estimates


    Q_current <- t(arsumg) %*% arcinv %*% arsumg                   #Current value of the quadratic inference function (QIF)


    if(iteration>0) QIFdiff = abs(Q_prev-Q_current)                #Convergence criterion

    bad_estimate=0      #Used to make sure the variable count has the correct value

    #Now to check that the Q decreased since the last iteration (assuming we are at least on the 2nd iteration)
    if(iteration>=1)
    {
      OK=0
      if( (Q_current>Q_prev) & (Q_prev<0) ) #If the QIF value becomes positive, or at least closer to 0, again after being negative (this is not likely), then keep these estimates.
      {
        OK=1
        bad_estimate=0
      }

      if( (Q_current>Q_prev | Q_current<0) & OK==0 )   #Checking if the QIF value has decreased. Also ensuring that the QIF is positive.  Rarely, it can be negative.
      {
        beta <- betaold - (.5^count)*invarqif2dev_old %*% arqif1dev_old
        count = count + 1
        Q_current = Q_prev
        QIFdiff=tol+1       #Make sure the code runs another iteration of the while loop
        betanew=beta
        bad_estimate=1      #used to make sure the variable count is not reset to 1
      }
    }

    if(bad_estimate==0)
    {
      betaold=beta                     #Saving current values for the previous code in the next iteration (if needed)
      invarqif2dev_old=invarqif2dev
      arqif1dev_old=arqif1dev
      count=1                          #count needs to be 1 if the QIF value is positive and has decreased
    }

    Q_prev = Q_current

    if(iteration==0) QIFdiff=tol+1 #Making sure at least two iterations are run

    iteration <- iteration + 1

  }


  #Updating the estimate of the correlation parameter
  scale=sum_scale/(N-np)

  #Here are the outer terms of the inverse of the asympototic covariance (so the derivative of the expected value of g)
  term = arsumgfirstdev

  #Here is the ordinary asymptotic empirical variance
  emp_var = invarqif2dev
  C_N_inv = arcinv                  #arcinv=ginv(C_N_working)
  C_N_original = C_N_working
  C_N_reg = S_Reg/nsub              #This is the typical non-weighted matrix, divided by nsub

  #Here is the estimate of g_N
  g_N = arsumg
  J_N = t(term) %*% C_N_inv %*% term / nsub


  #Obtaining G to correct/account for the covariance inflation (the for loop is used to obtain the derivative with respect to the empirical covariance matrix):

  loc1 <- 0
  loc2 <- 0
  for(i in 1:nsub){
    loc1 <- loc2 + 1
    loc2 <- loc1 + nobs[i] - 1
    yi <- as.matrix(y[loc1:loc2, ])
    xi <- x[loc1:loc2, ]
    ni <- nrow(yi)

    if (fam == "gaussian") {
      ui <- xi %*% beta_work
      fui <- ui
      fui_dev <- diag(ni)
      vui <- diag(ni)
    }
    if (fam == "poisson") {
      ui <- exp(xi %*% beta_work)
      fui <- log(ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui))
      if(ni==1)
        fui_dev <- diag(ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui)))
      if(ni==1)
        vui <- diag(sqrt(1/ui))
    }
    if (fam == "gamma") {
      ui <- 1/(xi %*% beta_work)
      fui <- 1/ui
      if(ni>1)
        fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
      if(ni==1)
        fui_dev <- -diag(ui) %*% diag(ui)
      if(ni>1)
        vui <- diag(as.vector(1/ui))
      if(ni==1)
        vui <- diag(1/ui)
    }
    if (fam == "binomial") {
      ui <- 1/(1 + exp(-xi %*% beta_work))
      fui <- log(ui) - log(1 - ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
      if(ni==1)
        fui_dev <- diag(ui) %*% diag(1 - ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
      if(ni==1)
        vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
    }

    Di = fui_dev %*% xi
    DiT = t(Di)

    gi <- matrix(0, ni * ni, np)
    for(j in 1:np) {
      if(type[j,1]==1){
        g_i <- Di[,j] %*% t(yi - ui)
        gi[,j] <- matrix(c(g_i))
      }
      if(type[j,1]==2){
        g_i <- Di[,j] %*% t(yi - ui)
        g_i[upper.tri(g_i)] <- NA
        gi[,j] <- matrix(c(g_i))
      }
      if(type[j,1]==3){
        g_i <- Di[,j] %*% t(yi - ui)
        g_i[lower.tri(g_i)] <- NA
        g_i[upper.tri(g_i)] <- NA
        gi[,j] <- matrix(c(g_i))
      }
    }

    gi1 <- matrix(c(gi))
    gi2 <- matrix(gi1[which(gi1!="NA")])

    d_i=list();i=1
    di=list();i=1
    for(j in 1:np){
      for(k in 1:np){
        d_i[[i]] <- - Di[,j] %*% t(t(Di)[k,])
        di[[i]] <- matrix(c(d_i[[i]]))
        i=i+1
      }
    }

    di1 <- matrix(do.call(cbind, di), ni * ni * np, np)
    for(l in 1:(ni*ni*np)){
      if (is.na(gi1[l,1]))
        di1[l,] <- NA
    }

    firstdev <- matrix(di1[which(di1!="NA")], length(gi2), np)

    #di2 <- matrix(0, length(gi2), np)
    #for(j in 1:np){
    #  di2[,j] <- matrix(c(di1[,j])[which(c(di1[,j])!="NA")])
    #}
    #firstdev <- di2

    G <- matrix(0, np, np)
    for(j in 1:np)
      G[,j] = G[,j] + (1-rho_N)*( ginv(J_N)%*%t(term)%*%(C_N_inv/nsub)%*%( firstdev[,j]%*%t(gi2) + gi2%*%t(firstdev[,j]) )%*%(C_N_inv/nsub)%*%g_N )/nsub
  }


  #Correcting for the other source of bias in the estimated asymptotic covariance matrix :


  loc1 <- 0
  loc2 <- 0
  for(i in 1:nsub){
    loc1 <- loc2 + 1
    loc2 <- loc1 + nobs[i] - 1
    yi <- as.matrix(y[loc1:loc2, ])
    xi <- x[loc1:loc2, ]
    ni <- nrow(yi)

    if (fam == "gaussian") {
      ui <- xi %*% beta
      fui <- ui
      fui_dev <- diag(ni)
      vui <- diag(ni)
    }
    if (fam == "poisson") {
      ui <- exp(xi %*% beta)
      fui <- log(ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui))
      if(ni==1)
        fui_dev <- diag(ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui)))
      if(ni==1)
        vui <- diag(sqrt(1/ui))
    }
    if (fam == "gamma") {
      ui <- 1/(xi %*% beta)
      fui <- 1/ui
      if(ni>1)
        fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
      if(ni==1)
        fui_dev <- -diag(ui) %*% diag(ui)
      if(ni>1)
        vui <- diag(as.vector(1/ui))
      if(ni==1)
        vui <- diag(1/ui)
    }
    if (fam == "binomial") {
      ui <- 1/(1 + exp(-xi %*% beta))
      fui <- log(ui) - log(1 - ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
      if(ni==1)
        fui_dev <- diag(ui) %*% diag(1 - ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
      if(ni==1)
        vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
    }

    Di = fui_dev %*% xi
    DiT = t(Di)
    wi = DiT %*% vui %*% vui

  }
  Model_basedVar = ginv(I_ind)

  max_length = np*(max(nobs))^2
  arsumc_NoBC1 <- matrix(0, max_length, max_length)
  arsumc_BC1 <- matrix(0, max_length, max_length)
  arsumc_BC2 <- matrix(0, max_length, max_length)

  loc1 <- 0
  loc2 <- 0
  for(i in 1:nsub){
    loc1 <- loc2 + 1
    loc2 <- loc1 + nobs[i] - 1
    yi <- as.matrix(y[loc1:loc2, ])
    xi <- x[loc1:loc2, ]
    ni <- nrow(yi)

    if (fam == "gaussian") {
      ui <- xi %*% beta
      fui <- ui
      fui_dev <- diag(ni)
      vui <- diag(ni)
    }
    if (fam == "poisson") {
      ui <- exp(xi %*% beta)
      fui <- log(ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui))
      if(ni==1)
        fui_dev <- diag(ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui)))
      if(ni==1)
        vui <- diag(sqrt(1/ui))
    }
    if (fam == "gamma") {
      ui <- 1/(xi %*% beta)
      fui <- 1/ui
      if(ni>1)
        fui_dev <- -diag(as.vector(ui)) %*% diag(as.vector(ui))
      if(ni==1)
        fui_dev <- -diag(ui) %*% diag(ui)
      if(ni>1)
        vui <- diag(as.vector(1/ui))
      if(ni==1)
        vui <- diag(1/ui)
    }
    if (fam == "binomial") {
      ui <- 1/(1 + exp(-xi %*% beta))
      fui <- log(ui) - log(1 - ui)
      if(ni>1)
        fui_dev <- diag(as.vector(ui)) %*% diag(as.vector(1 - ui))
      if(ni==1)
        fui_dev <- diag(ui) %*% diag(1 - ui)
      if(ni>1)
        vui <- diag(as.vector(sqrt(1/ui))) %*% diag(as.vector(sqrt(1/(1 - ui))))
      if(ni==1)
        vui <- diag(sqrt(1/ui)) %*% diag(sqrt(1/(1 - ui)))
    }

    Di = fui_dev %*% xi
    DiT = t(Di)
    wi = DiT %*% vui %*% vui


    #Correction similar to the one proposed by Mancl and DeRouen (2001) with GEE:

    Ident = diag(ni)
    Hi = Di %*% Model_basedVar %*% wi

    gi_NoBC = matrix(0, ni * ni, np)
    gi_NoBC2 = matrix(0, ni * ni, np)
    gi_BC_1 = matrix(0, ni * ni, np)
    gi_BC2_1 = matrix(0, ni * ni, np)
    gi_BC_2 = matrix(0, ni * ni, np)
    gi_BC2_2 = matrix(0, ni * ni, np)

    for(j in 1:np)
    {
      if(type[j,1]==1)
      {
        g_i_NoBC <- (1/nsub) * Di[,j] %*% t(yi - ui)
        gi_NoBC[,j] <- matrix(c(g_i_NoBC))

        g_i_NoBC2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        gi_NoBC2[,j] <- matrix(c(g_i_NoBC2))

        g_i_BC_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        gi_BC_1[,j] <- matrix(c(g_i_BC_1))

        g_i_BC2_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        gi_BC2_1[,j] <- matrix(c(g_i_BC2_1))

        g_i_BC_2 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        gi_BC_2[,j] <- matrix(c(g_i_BC_2))

        g_i_BC2_2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        gi_BC2_2[,j] <- matrix(c(g_i_BC2_2))
      }
      if(type[j,1]==2)
      {
        g_i_NoBC <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_NoBC[upper.tri(g_i_NoBC)] <- NA
        gi_NoBC[,j] <- matrix(c(g_i_NoBC))

        g_i_NoBC2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_NoBC2[upper.tri(g_i_NoBC2)] <- NA
        gi_NoBC2[,j] <- matrix(c(g_i_NoBC2))

        g_i_BC_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC_1[upper.tri(g_i_BC_1)] <- NA
        gi_BC_1[,j] <- matrix(c(g_i_BC_1))

        g_i_BC2_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC2_1[upper.tri(g_i_BC2_1)] <- NA
        gi_BC2_1[,j] <- matrix(c(g_i_BC2_1))

        g_i_BC_2 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC_2[upper.tri(g_i_BC_2)] <- NA
        gi_BC_2[,j] <- matrix(c(g_i_BC_2))

        g_i_BC2_2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_BC2_2[upper.tri(g_i_BC2_2)] <- NA
        gi_BC2_2[,j] <- matrix(c(g_i_BC2_2))
      }
      if(type[j,1]==3)
      {
        g_i_NoBC <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_NoBC[lower.tri(g_i_NoBC)] <- NA
        g_i_NoBC[upper.tri(g_i_NoBC)] <- NA
        gi_NoBC[,j] <- matrix(c(g_i_NoBC))

        g_i_NoBC2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_NoBC2[lower.tri(g_i_NoBC2)] <- NA
        g_i_NoBC2[upper.tri(g_i_NoBC2)] <- NA
        gi_NoBC2[,j] <- matrix(c(g_i_NoBC2))

        g_i_BC_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC_1[lower.tri(g_i_BC_1)] <- NA
        g_i_BC_1[upper.tri(g_i_BC_1)] <- NA
        gi_BC_1[,j] <- matrix(c(g_i_BC_1))

        g_i_BC2_1 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC2_1[lower.tri(g_i_BC2_1)] <- NA
        g_i_BC2_1[upper.tri(g_i_BC2_1)] <- NA
        gi_BC2_1[,j] <- matrix(c(g_i_BC2_1))

        g_i_BC_2 <- (1/nsub) * Di[,j] %*% t(yi - ui) %*% ginv(Ident - Hi)
        g_i_BC_2[lower.tri(g_i_BC_2)] <- NA
        g_i_BC_2[upper.tri(g_i_BC_2)] <- NA
        gi_BC_2[,j] <- matrix(c(g_i_BC_2))

        g_i_BC2_2 <- (1/nsub) * Di[,j] %*% t(yi - ui)
        g_i_BC2_2[lower.tri(g_i_BC2_2)] <- NA
        g_i_BC2_2[upper.tri(g_i_BC2_2)] <- NA
        gi_BC2_2[,j] <- matrix(c(g_i_BC2_2))
      }
    }
    gi1_NoBC <- matrix(c(gi_NoBC))
    gi2_NoBC <- matrix(gi1_NoBC[which(gi1_NoBC!="NA")])

    gi1_NoBC2 <- matrix(c(gi_NoBC2))
    gi2_NoBC2 <- matrix(gi1_NoBC2[which(gi1_NoBC2!="NA")])

    gi1_BC_1 <- matrix(c(gi_BC_1))
    gi2_BC_1 <- matrix(gi1_BC_1[which(gi1_BC_1!="NA")])

    gi1_BC2_1 <- matrix(c(gi_BC2_1))
    gi2_BC2_1 <- matrix(gi1_BC2_1[which(gi1_BC2_1!="NA")])

    gi1_BC_2 <- matrix(c(gi_BC_2))
    gi2_BC_2 <- matrix(gi1_BC_2[which(gi1_BC_2!="NA")])

    gi1_BC2_2 <- matrix(c(gi_BC2_2))
    gi2_BC2_2 <- matrix(gi1_BC2_2[which(gi1_BC2_2!="NA")])

    #No Correction (Obtaining uncorrected empirical covariances in C_N using the final parameter estimates):
    arsumc_NoBC1 <- arsumc_NoBC1 + gi1_NoBC %*% t(gi1_NoBC2)
    arsumc_NoBC <- matrix(arsumc_NoBC1[which(arsumc_NoBC1!="NA")], length(gi2_BC_1), length(gi2_BC2_1))

    #Mancl and DeRouen (2001) with (I-H):
    arsumc_BC1 <- arsumc_BC1 + gi1_BC_1 %*% t(gi1_BC2_1)
    arsumc_BC_1 <- matrix(arsumc_BC1[which(arsumc_BC1!="NA")], length(gi2_BC_1), length(gi2_BC2_1))

    #Kauermann and Carroll (2001) with (I-H):
    arsumc_BC2 <- arsumc_BC2 + gi1_BC_2 %*% t(gi1_BC2_2)
    arsumc_BC_2 <- matrix(arsumc_BC2[which(arsumc_BC2!="NA")], length(gi2_BC_2), length(gi2_BC2_2))

  } #The end of the for loop, iterating through all nsub subjects


  #Here is the uncorrected sandwich covariance:
  sand_var = emp_var %*% ( t(term) %*% C_N_inv %*% arsumc_NoBC %*% C_N_inv %*% term ) %*% emp_var

  #Here are the bias-corrected covariance estimates:
  covariance_MD = (diag(np)+G) %*% emp_var %*% t(term) %*% C_N_inv %*% arsumc_BC_1 %*% C_N_inv %*% term %*% emp_var %*% t(diag(np)+G)
  covariance_KC = (diag(np)+G) %*% emp_var %*% t(term) %*% C_N_inv %*% arsumc_BC_2 %*% C_N_inv %*% term %*% emp_var %*% t(diag(np)+G)
  covariance_AVG = (covariance_MD + covariance_KC)/2

  I_ind = I_ind/scale

  beta_number=as.matrix(seq(1:np))-1 #Used to indicate the parameter. For instance, 0 denotes the intercept.
  SE_A=SE_MD=SE_KC=SE_AVG=TECM_A=TECM_MD=TECM_KC=TECM_AVG=CIC_A=CIC_MD=CIC_KC=CIC_AVG=matrix(0,np,1)
  for(j in 1:np){
    SE_A[j,1]=sqrt(sand_var[j,j])
    SE_MD[j,1]=sqrt(covariance_MD[j,j])
    SE_KC[j,1]=sqrt(covariance_KC[j,j])
    SE_AVG[j,1]=sqrt(covariance_AVG[j,j])}
  Wald_A=Wald_MD=Wald_KC=Wald_AVG=matrix(0,np,1)
  for(j in 1:np){
    Wald_A[j,1]=beta[j,1]/SE_A[j,1]
    Wald_MD[j,1]=beta[j,1]/SE_MD[j,1]
    Wald_KC[j,1]=beta[j,1]/SE_KC[j,1]
    Wald_AVG[j,1]=beta[j,1]/SE_AVG[j,1]}
  df=nsub-np
  p_valueA=p_valueMD=p_valueKC=p_valueAVG=matrix(0,np,1)
  for(j in 1:np){
    p_valueA[j,1]=1-pf((Wald_A[j,1])^2,1,df)
    p_valueMD[j,1]=1-pf((Wald_MD[j,1])^2,1,df)
    p_valueKC[j,1]=1-pf((Wald_KC[j,1])^2,1,df)
    p_valueAVG[j,1]=1-pf((Wald_AVG[j,1])^2,1,df)}

  TECM_A[1,1]=sum(diag( sand_var ))
  TECM_MD[1,1]=sum(diag( covariance_MD ))
  TECM_KC[1,1]=sum(diag( covariance_KC ))
  TECM_AVG[1,1]=sum(diag( covariance_KC ))
  CIC_A[1,1]=sum(diag( I_ind %*% sand_var ))
  CIC_MD[1,1]=sum(diag( I_ind %*% covariance_MD ))
  CIC_KC[1,1]=sum(diag( I_ind %*% covariance_KC ))
  CIC_AVG[1,1]=sum(diag( I_ind %*% covariance_AVG ))

  output_A=cbind(beta_number,beta,SE_A,Wald_A,p_valueA,TECM_A,CIC_A)
  output_MD=cbind(beta_number,beta,SE_MD,Wald_MD,p_valueMD,TECM_MD,CIC_MD)
  output_KC=cbind(beta_number,beta,SE_KC,Wald_KC,p_valueKC,TECM_KC,CIC_KC)
  output_AVG=cbind(beta_number,beta,SE_AVG,Wald_AVG,p_valueAVG,TECM_AVG,CIC_AVG)

  colnames(output_A) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_MD) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_KC) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")
  colnames(output_AVG) <- c("Parameter", "Estimate", "Standard Error", "Wald Statistic", "p-value", "TECM", "CIC")

  correction_A=correction_MD=correction_KC=correction_AVG=matrix(0,1,7)
  correction_A[1,]=c("Typical Asymptotic SE","","","","","","")
  correction_MD[1,]=c("Bias-Corrected SE (MD)","","","","","","")
  correction_KC[1,]=c("Bias-Corrected SE (KC)","","","","","","")
  correction_AVG[1,]=c("Bias-Corrected SE (AVG)","","","","","","")
  output_ALL=rbind(correction_A,output_A,correction_MD,output_MD,correction_KC,output_KC,correction_AVG,output_AVG)

  message("Results include typical asymptotic standard error (SE) estimates and bias-corrected SE estimates that use covariance inflation corrections.")
  message("Covariance inflation corrections are based on the corrections of Mancl and DeRouen (MD) (2001), and Kauermann and Carroll (KC) (2001).")
  message("Last covariance inflation correction averages the above corrections (AVG) (Ford and Westgate, 2017, 2018).")
  return(knitr::kable(output_ALL))
}
