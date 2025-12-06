import azure.functions as func
import json
from recommander.recommander_main import get_recommendations

app = func.FunctionApp()

##########################################
# Azure Function : API de recommandation #
##########################################
@app.function_name(name="recommande")
@app.route(route="recommande", methods=["GET"],
           auth_level=func.AuthLevel.ANONYMOUS)
def recommander(req: func.HttpRequest) -> func.HttpResponse:
    try:
        userID_param = req.params.get("user_id")
        nbReco_param = req.params.get("nb_reco", "5")
        if(userID_param is None):
            return func.HttpResponse("Paramètre \"user_id\" obligatoire !", status_code=400)
        user_id = int(userID_param)
        nb_reco = int(nbReco_param)
        resultat = get_recommendations(user_id=user_id, nbReco=nb_reco)
        return func.HttpResponse(json.dumps(resultat, ensure_ascii=False, indent=4), mimetype="application/json", status_code=200)
    except Exception as expt:
        return func.HttpResponse(json.dumps({"erreur": str(expt)}), mimetype="application/json", status_code=500)