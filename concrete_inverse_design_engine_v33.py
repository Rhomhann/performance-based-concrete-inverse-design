
"""
Performance-Based Concrete Inverse Design Engine V3.2

Key correction:
- Binder system is enforced DURING candidate generation.
- Cement-only means Slag = 0 and Fly Ash = 0 before optimization.
- Project constraints are hard constraints during search.
- If no supported solution is found, the engine returns an engineering diagnostic
  rather than claiming the concrete itself is physically impossible.

Search/ranking: Gradient Boosting + Random Forest
Independent computational audit: Extra Trees
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

MIX_COLS = ["Cement","Furnace_Slag","Fly_Ash","Water","Admixture","C_Agg","F_Agg"]
FEATURES = MIX_COLS + ["Age"]

class DesignDomainError(RuntimeError):
    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}

class ConcreteInverseDesignEngineV33:
    def __init__(self, data_csv, random_state=1234):
        self.random_state=random_state
        d=pd.read_csv(data_csv)
        d.columns=MIX_COLS+["Age","Strength"]
        self.data=d[(d["Age"]>6.9)&(d["Age"]<28.5)].reset_index(drop=True)

        dev,self.locked_test=train_test_split(self.data,test_size=.20,random_state=random_state)
        self.train,self.validation=train_test_split(dev,test_size=.20,random_state=random_state)

        self.guide=GradientBoostingRegressor(
            random_state=random_state,n_estimators=350,max_depth=2,
            learning_rate=.035,loss="huber")
        self.search_model=RandomForestRegressor(
            n_estimators=600,random_state=4321,n_jobs=-1,max_features=.9)
        self.audit_model=ExtraTreesRegressor(
            n_estimators=600,random_state=random_state,n_jobs=-1)

        for m in (self.guide,self.search_model,self.audit_model):
            m.fit(self.train[FEATURES],self.train["Strength"])

        self.lo=self.train[MIX_COLS].min().to_numpy(float)
        self.hi=self.train[MIX_COLS].max().to_numpy(float)
        self.sd=np.maximum(self.train[MIX_COLS].std().to_numpy(float),1.0)

        inp=self.train[["Age","Strength"]].to_numpy(float)
        self.input_mu=inp.mean(0)
        self.input_sd=np.maximum(inp.std(0),1e-9)
        self.nn=NearestNeighbors(n_neighbors=5).fit((inp-self.input_mu)/self.input_sd)

    def _rng_for(self,age,target,binder_system):
        key=f"{self.random_state}|{age:.6f}|{target:.6f}|{binder_system}".encode()
        return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:8],16))

    def _compatible_training(self,binder_system):
        if binder_system=="Cement only":
            return self.train[(self.train["Furnace_Slag"]==0)&(self.train["Fly_Ash"]==0)].copy()
        if binder_system=="No fly ash":
            return self.train[self.train["Fly_Ash"]==0].copy()
        if binder_system=="No slag":
            return self.train[self.train["Furnace_Slag"]==0].copy()
        return self.train.copy()

    def _seed_mix(self,age,target,reference):
        X=reference[["Age","Strength"]].to_numpy(float)
        mu=X.mean(0); sd=np.maximum(X.std(0),1e-9)
        z=(X-mu)/sd; q=(np.array([age,target])-mu)/sd
        dist=np.sqrt(((z-q)**2).sum(1))
        k=min(12,len(dist))
        ids=np.argpartition(dist,k-1)[:k]
        w=1/np.maximum(dist[ids],1e-6); w/=w.sum()
        return np.sum(reference.iloc[ids][MIX_COLS].to_numpy(float)*w[:,None],0)

    def _research_feasible(self,c):
        binder=c[:,0]+c[:,1]+c[:,2]
        wcm=c[:,3]/np.maximum(binder,1e-9)
        total=c.sum(1)
        return ((c>=0).all(1)&(wcm>=.25)&(wcm<=.65)&
                (binder>=250)&(binder<=650)&(total>=2100)&(total<=2600))

    @staticmethod
    def _n(v):
        if v is None or v=="": return None
        return float(v)

    def _constraint_mask(self,c,constraints):
        binder=c[:,0]+c[:,1]+c[:,2]
        wcm=c[:,3]/np.maximum(binder,1e-9)
        slag=100*c[:,1]/np.maximum(binder,1e-9)
        fly=100*c[:,2]/np.maximum(binder,1e-9)
        scm=100*(c[:,1]+c[:,2])/np.maximum(binder,1e-9)
        keep=np.ones(len(c),dtype=bool)
        tests=[]
        def apply(name,mask):
            nonlocal keep
            before=int(keep.sum())
            keep &= mask
            tests.append((name,before,int(keep.sum())))

        bs=constraints.get("binder_system","SCM permitted")
        if bs=="Cement only":
            apply("Cement-only binder (slag = 0; fly ash = 0)",
                  (np.isclose(c[:,1],0))&(np.isclose(c[:,2],0)))
        elif bs=="No fly ash":
            apply("No fly ash",np.isclose(c[:,2],0))
        elif bs=="No slag":
            apply("No slag",np.isclose(c[:,1],0))

        v=self._n(constraints.get("max_wcm"))
        if v is not None: apply(f"Maximum w/cm ≤ {v:g}",wcm<=v)
        lo=self._n(constraints.get("min_binder"))
        hi=self._n(constraints.get("max_binder"))
        if lo is not None: apply(f"Minimum binder ≥ {lo:g} kg/m³",binder>=lo)
        if hi is not None: apply(f"Maximum binder ≤ {hi:g} kg/m³",binder<=hi)
        v=self._n(constraints.get("max_slag_pct"))
        if v is not None: apply(f"Slag ≤ {v:g}% of binder",slag<=v)
        v=self._n(constraints.get("max_flyash_pct"))
        if v is not None: apply(f"Fly ash ≤ {v:g}% of binder",fly<=v)
        v=self._n(constraints.get("max_total_scm_pct"))
        if v is not None: apply(f"Total SCM ≤ {v:g}% of binder",scm<=v)
        lo=self._n(constraints.get("min_cement"))
        hi=self._n(constraints.get("max_cement"))
        if lo is not None: apply(f"Minimum cement ≥ {lo:g} kg/m³",c[:,0]>=lo)
        if hi is not None: apply(f"Maximum cement ≤ {hi:g} kg/m³",c[:,0]<=hi)
        return keep,tests

    def _diagnostic(self,c,constraints,target,age,reference,tests):
        d={"target":target,"age":age,"binder_system":constraints.get("binder_system","SCM permitted")}
        d["compatible_training_rows"]=len(reference)
        local=reference[(reference["Age"].sub(age).abs()<=3)&
                        (reference["Strength"].sub(target).abs()<=5)]
        d["compatible_local_support"]=len(local)
        d["constraint_attrition"]=tests

        if len(c):
            binder=c[:,0]+c[:,1]+c[:,2]
            wcm=c[:,3]/np.maximum(binder,1e-9)
            d["candidate_wcm_range"]=(float(wcm.min()),float(wcm.max()))
            d["candidate_binder_range"]=(float(binder.min()),float(binder.max()))

        d["interpretation"]=(
            "No supported candidate was identified within the current learned design domain "
            "under all selected requirements. This does not establish that the requested concrete "
            "is physically impossible."
        )
        if len(reference)==0:
            d["reason"]="The training subset contains no mixtures compatible with the selected binder system."
        elif len(local)<3:
            d["reason"]=(
                f"The selected binder system has sparse local data near {target:g} MPa at {age:g} days "
                f"({len(local)} training mixtures within ±5 MPa and ±3 days), so recommendations are weakly supported."
            )
        elif tests and tests[-1][2]==0:
            limiting=next((x for x in tests if x[2]==0),tests[-1])
            d["reason"]=f"The first requirement that eliminated the remaining candidate pool was: {limiting[0]}."
        else:
            d["reason"]="The combined requirements eliminated the candidate search pool."
        d["recommended_action"]=(
            "Review the binding project requirement(s), expand data for the requested binder system, "
            "or proceed to engineer-directed trial mixture development using the actual project materials."
        )
        return d

    def design(self,target_strength_mpa,age_days,n_candidates=18000,top_n=3,constraints=None):
        target,age=float(target_strength_mpa),float(age_days)
        constraints=constraints or {}
        binder_system=constraints.get("binder_system","SCM permitted")
        rng=self._rng_for(age,target,binder_system)
        warnings=[]

        reference=self._compatible_training(binder_system)
        if len(reference)<5:
            raise DesignDomainError(
                "Insufficient compatible training data for the selected binder system.",
                {"compatible_training_rows":len(reference),
                 "reason":"Too few compatible mixtures are represented in the learned domain.",
                 "interpretation":"This is a data-support limitation, not proof of physical infeasibility."})

        seed=np.clip(self._seed_mix(age,target,reference),self.lo,self.hi)

        # Generate around the binder-compatible seed.
        local=seed+rng.normal(0,.22*self.sd,size=(n_candidates,7))
        ex=reference[MIX_COLS].sample(
            n=min(2500,max(500,len(reference)*5)),replace=True,
            random_state=int(rng.integers(0,2**31-1))).to_numpy(dtype=float, copy=True)
        ex = ex + rng.normal(0,.06*self.sd,size=ex.shape)

        cand=np.vstack([seed,local,ex])
        cand=np.clip(cand,self.lo,self.hi)

        # HARD binder-system enforcement occurs BEFORE feasibility/optimization.
        if binder_system=="Cement only":
            cand[:,1]=0.0; cand[:,2]=0.0
        elif binder_system=="No fly ash":
            cand[:,2]=0.0
        elif binder_system=="No slag":
            cand[:,1]=0.0

        cand=cand[self._research_feasible(cand)]
        n_research=len(cand)
        if not len(cand):
            raise DesignDomainError(
                "No candidate survived the preliminary research feasibility screen.",
                self._diagnostic(cand,constraints,target,age,reference,[]))

        keep,tests=self._constraint_mask(cand,constraints)
        filtered=cand[keep]
        if not len(filtered):
            raise DesignDomainError(
                "No supported candidate was identified under all selected project requirements.",
                self._diagnostic(cand,constraints,target,age,reference,tests))
        cand=filtered
        n_project=len(cand)

        X=pd.DataFrame(np.column_stack([cand,np.full(len(cand),age)]),columns=FEATURES)
        pg=self.guide.predict(X); ps=self.search_model.predict(X)
        consensus=(pg+ps)/2
        disagreement=np.abs(pg-ps)
        seed_distance=np.mean(((cand-seed)/self.sd)**2,axis=1)
        score=np.abs(consensus-target)+.35*disagreement+.18*seed_distance
        order=np.argsort(score)

        chosen=[]
        for j in order:
            if not chosen or all(np.sqrt(np.mean(((cand[j]-cand[k])/self.sd)**2))>.12 for k in chosen):
                chosen.append(j)
            if len(chosen)==top_n: break
        if not chosen:
            raise DesignDomainError("No distinct candidate could be selected.")

        audit=self.audit_model.predict(X.iloc[chosen])

        # Binder-system-specific local support is more informative than global support.
        local_support=int(((reference["Age"]-age).abs()<=3).mul(
                          (reference["Strength"]-target).abs()<=5).sum())
        ref_in=reference[["Age","Strength"]].to_numpy(float)
        mu=ref_in.mean(0); sd=np.maximum(ref_in.std(0),1e-9)
        nn=NearestNeighbors(n_neighbors=min(5,len(reference))).fit((ref_in-mu)/sd)
        ad=float(nn.kneighbors((np.array([[age,target]])-mu)/sd)[0].mean())

        outside=not (self.data["Age"].min()<=age<=self.data["Age"].max() and
                     self.data["Strength"].min()<=target<=self.data["Strength"].max())

        rows=[]
        for rank,(j,pa) in enumerate(zip(chosen,audit),1):
            b=cand[j,0]+cand[j,1]+cand[j,2]
            ae=abs(pa-target)
            if not outside and ae<=3 and disagreement[j]<=3 and local_support>=5 and ad<=1:
                conf="High"
            elif not outside and ae<=5 and disagreement[j]<=5 and local_support>=2 and ad<=1.5:
                conf="Moderate"
            else: conf="Low"
            rows.append({
                "Rank":rank,"Binder_System":binder_system,
                "Cement_kg_m3":cand[j,0],"Slag_kg_m3":cand[j,1],"Fly_Ash_kg_m3":cand[j,2],
                "Water_kg_m3":cand[j,3],"Admixture_kg_m3":cand[j,4],
                "Coarse_Aggregate_kg_m3":cand[j,5],"Fine_Aggregate_kg_m3":cand[j,6],
                "Water_Binder_Ratio":cand[j,3]/b,"Total_Binder_kg_m3":b,
                "Slag_pct_of_Binder":100*cand[j,1]/b,
                "Fly_Ash_pct_of_Binder":100*cand[j,2]/b,
                "Total_SCM_pct_of_Binder":100*(cand[j,1]+cand[j,2])/b,
                "Guide_Strength_MPa":pg[j],"Search_Model_Strength_MPa":ps[j],
                "Search_Consensus_MPa":consensus[j],
                "Search_Model_Disagreement_MPa":disagreement[j],
                "Independent_Audit_Strength_MPa":pa,
                "Audit_Error_MPa":pa-target,"Absolute_Audit_Error_MPa":ae,
                "Compatible_Local_Training_Support_n":local_support,
                "Applicability_5NN_Distance":ad,"Confidence":conf,
                "Research_Feasible_Pool_n":n_research,
                "Project_Constrained_Pool_n":n_project,
                "Constraint_Basis":constraints.get("basis","Project specification / engineer-defined requirement")
            })
        result=pd.DataFrame(rows)

        # PERFORMANCE-ACHIEVEMENT GATE
        # This is deliberately separate from model confidence/applicability.
        # A compositionally feasible mixture is not presented as a successful
        # performance-based design if the independent computational audit misses
        # the requested strength materially.
        result["Performance_Status"] = np.select(
            [
                result["Absolute_Audit_Error_MPa"] <= 3.0,
                result["Absolute_Audit_Error_MPa"] <= 5.0
            ],
            [
                "Computational target satisfied",
                "Marginal — engineering review required"
            ],
            default="Target not achieved in current supported search"
        )
        result["Trial_Batch_Recommendation"] = np.select(
            [
                result["Absolute_Audit_Error_MPa"] <= 3.0,
                result["Absolute_Audit_Error_MPa"] <= 5.0
            ],
            [
                "Candidate for trial batching",
                "Conditional candidate — engineering review before trial batching"
            ],
            default="Do not present as a target-achieving candidate"
        )

        if binder_system=="Cement only":
            warnings.append(
                f"Cement-only mode enforced slag = 0 and fly ash = 0 during generation. "
                f"Compatible training subset: {len(reference)} rows; local support near target: {local_support}.")
        if local_support<5:
            warnings.append("Sparse binder-system-specific support near the requested performance; interpret recommendations cautiously.")
        if n_project<max(50,int(.01*n_research)):
            warnings.append("Project requirements leave a small candidate pool; the solution is constraint-sensitive.")
        warnings.append("Independent audit is computational ML verification, not laboratory validation.")
        return result,warnings,tests
