function fn_GoCommentPage2(page) {
	
	if(commentPageProgress) {
		return;
	}
	commentPageProgress = true;
	
	var pFilterVal = $("#commentFilterCol").val(); // ���簪 (���ͱ���)
	var sortCol = $("#commentSort").val(); // ���簪 (���ı���)
	var commentItemId = $("#commentItemId").val(); // ���簪 (commentItemId)
	var commentSiteNo = $("#commentSiteNo").val(); // ���簪 (����Ʈ �ѹ�)
	var commentUitemId = $("#commentUitemId").val(); // ���簪 (all)
	if( commentUitemId === 'all' ){
		commentUitemId = '';
	}
	var recomAttrGrpId = $("#recomAttrGrpId").val(); // ���簪
	var recomAttrId = $("#recomAttrId").val(); // ���簪 ("")
	var commentOreItemId = $("#commentOreItemId").val(); // ���簪 ("")
	var commentOreItemReviewYn = $("#commentOreItemReviewYn").val(); // ���簪("N")

	var checkCnt = 0;

	if( recomAttrGrpId !== '' && recomAttrId !== '' ){
		pFilterVal = '10';
	}
	
	$("#commentPage").val(page);
	var url = "/item/ajaxItemCommentList.ssg";
	var paramData = {itemId:commentItemId, siteNo:commentSiteNo, filterCol:pFilterVal, sortCol:sortCol, uitemId:commentUitemId, recomAttrGrpId:recomAttrGrpId, recomAttrId:recomAttrId, page:page, pageSize:"10", oreItemId:commentOreItemId, oreItemReviewYn:commentOreItemReviewYn};

	$.ajax({
		type: "GET",
		url: url,
		data: paramData,
		dataType: "json",
		success: function (data) {
			
			console.log(data);
			return data;
			
		},
		error: function () {
			alert("����");
			commentPageProgress = false;
		}
	});
}